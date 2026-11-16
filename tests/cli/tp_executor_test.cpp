// Drives the real scheduler and runner pool on rank 0 through
// `TpMirroredRunner`, and a `TpExecutor` on rank 1, over a loopback control
// channel. A toy model records every model call each rank makes; the pair is
// correct when both ranks record the same calls with the same results, for
// greedy, sampled, stopped, cancelled, multi-token, concurrent and batched
// requests, when rank 1 reports the divergences it is meant to catch, and when
// a request that reused what rank 1 then rejected fails too.

#include "src/cli/serve/tp_executor.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/tp_bulk.hpp"
#include "src/cli/serve/tp_control.hpp"
#include "tests/cache/fake_adapter.hpp"
#include "tests/cli/fake_cache_resources.hpp"

namespace {

using gufo::server::ChatRequest;
using gufo::server::TextDecodeSelection;
using gufo::server::TextDecodeStep;
using gufo::server::TextExecutionPlan;
using gufo::server::TextExecutionPlanKind;
using gufo::server::TextGenerationBackend;
using gufo::server::TextGenerationScheduler;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillStep;
using gufo::server::TextRequestMetadata;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerPool;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;
using gufo::server::TpBulkHeader;
using gufo::server::TpBulkKind;
using gufo::server::TpBulkSocket;
using gufo::server::TpCallScope;
using gufo::server::TpControlChannel;
using gufo::server::TpControlCommand;
using gufo::server::TpControlCommandKind;
using gufo::server::TpControlConfig;
using gufo::server::TpControlInstructionSink;
using gufo::server::TpControlResponse;
using gufo::server::TpExecutor;
using gufo::server::TpInstruction;
using gufo::server::TpInstructionOp;
using gufo::server::TpInstructionSink;
using gufo::server::TpMirroredRunner;
using gufo::server::TpRequestContext;
using FinishReason = TextGenerationBackend::FinishReason;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

std::uint16_t FreePort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
  socklen_t length = sizeof(address);
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length);
  const auto port = ntohs(address.sin_port);
  ::close(fd);
  return port;
}

constexpr std::size_t kVocab = 32;
constexpr TextRunnerToken kEos = 0;

/// One model call and what it produced, as a comparable line.
using CallLog = std::vector<std::string>;

struct ToyOptions {
  bool mtp{false};
  bool cache{false};
  /// Fail the next private-state capture or load, like a failed transfer.
  bool fail_capture_once{false};
  bool fail_restore_once{false};
  /// Row bytes the cache can retain; small values force eviction.
  std::size_t cache_row_bytes{std::size_t{1} << 24};
  std::function<void()> capture_hook;
  /// Runs before every prefill chunk.
  std::function<void()> prefill_hook;
  std::size_t prefill_chunk{3};
  /// Prefill continues a prompt exactly only from multiples of this.
  std::size_t prefill_alignment{1};
  /// Widest batched plan offered; one offers only serial steps. With `mtp`,
  /// multi-token steps are batched too.
  std::size_t width{1};
  /// The draft count a greedy multi-token batch chooses on this rank, like
  /// Flash-Next's timing-based choice: the ranks may choose differently.
  std::optional<std::uint32_t> batch_plan;
  /// Rank-1 faults: shift the greedy choice once, at this sequence length.
  std::optional<std::size_t> diverge_at;
  /// Rank-1 fault: report one extra draft in the next multi-token step.
  bool extra_draft_once{false};
  /// Rank-1 fault: refuse to rebuild a request's prompt context.
  bool fail_context_decode{false};
  /// Offer structured output. No toy piece is JSON, so a constraint refuses
  /// every token.
  bool constraint{false};
  /// Reported instead of the capabilities the options above imply.
  std::optional<TextRunnerCapabilities> capabilities;
};

/// A toy "image": it shifts every logit of the sequence that sees it, and its
/// shade is its cache identity.
struct ToyContext final : gufo::server::TextPromptContext {
  explicit ToyContext(std::uint8_t shade) : shade(shade) {
    cache_identity = {shade};
  }
  std::uint8_t shade;
};

class ToyRunner;

/// The cache adapter of one rank, with injected transfer faults.
class ToyAdapter final : public gufo::cache::testing::FakeAdapter {
public:
  ToyAdapter(ToyOptions& options, std::function<void()> on_invalidate)
      : FakeAdapter(64),
        options_(options),
        on_invalidate_(std::move(on_invalidate)) {}
  /// Resets come from the cache or a state; both are the same model call.
  bool Invalidate(gufo::cache::Slot& slot) noexcept override {
    try {
      on_invalidate_();
    } catch (...) {
    }
    return FakeAdapter::Invalidate(slot);
  }
  gufo::cache::Completion CapturePrivatePiece(
      const gufo::cache::Slot& slot, gufo::cache::ComponentId id,
      std::size_t offset, std::span<std::byte> bytes,
      gufo::cache::Stream& stream) override {
    if (offset == 0) {
      if (options_.capture_hook)
        options_.capture_hook();
      if (std::exchange(options_.fail_capture_once, false))
        FailNextTransfer();
    }
    return FakeAdapter::CapturePrivatePiece(slot, id, offset, bytes, stream);
  }
  gufo::cache::Completion LoadPrivatePiece(
      gufo::cache::Slot& slot, gufo::cache::ComponentId id, std::size_t offset,
      std::span<const std::byte> bytes, gufo::cache::Stream& stream) override {
    if (offset == 0 && std::exchange(options_.fail_restore_once, false))
      FailNextTransfer();
    return FakeAdapter::LoadPrivatePiece(slot, id, offset, bytes, stream);
  }

private:
  ToyOptions& options_;
  std::function<void()> on_invalidate_;
};

class ToyState final : public TextRunnerState {
public:
  ToyState(const ToyRunner* owner, ToyAdapter& adapter, gufo::cache::Slot& slot,
           std::size_t id)
      : owner_(owner), adapter(adapter), slot(slot), id_(id) {}
  void Invalidate() noexcept override;
  void SetCancellationCheck(const CancellationCheck&) override;

  std::vector<TextRunnerToken> tokens;
  /// The bound context's shade, zero without one.
  std::uint8_t shade{0};
  std::size_t id() const { return id_; }

private:
  const ToyRunner* owner_;

public:
  ToyAdapter& adapter;
  gufo::cache::Slot& slot;

private:
  std::size_t id_;
};

/// A deterministic model whose next token depends on the whole sequence, so
/// any divergence between the ranks changes later tokens too.
class ToyRunner final : public TextModelRunner {
public:
  explicit ToyRunner(ToyOptions options) : options_(options) {}
  DeviceProbeStatus probe_status{DeviceProbeStatus::kUsable};
  DeviceProbeStatus PollDevice() const override { return probe_status; }
  bool DeviceUsable() const override {
    return probe_status != DeviceProbeStatus::kLost;
  }

  CallLog Log() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return log_;
  }
  void Record(std::string line) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    log_.push_back(std::move(line));
  }
  void FailNextRestore() { options_.fail_restore_once = true; }
  std::size_t prefill_calls() const { return prefill_calls_.load(); }
  std::size_t cancellation_checks() const { return cancellation_checks_; }
  void CountCancellationCheck() const { ++cancellation_checks_; }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "toy",
            .state_abi = "toy-v1",
            .max_context = 256,
            .capabilities =
                options_.capabilities
                    ? *options_.capabilities
                    : TextRunnerCapabilities{
                          .incremental_prefill = true,
                          .final_token_advance_required = false,
                          .incremental_text_is_exact = true,
                          .multi_token_decode = options_.mtp,
                          .batched_multi_token_decode =
                              options_.mtp && options_.width > 1,
                          .batched_multi_token_decode_max_width =
                              options_.mtp ? options_.width : 0,
                          .prefix_reuse = options_.cache,
                          .prefill_alignment = options_.prefill_alignment,
                      },
            .scheduling = {.pass_tokens = 512}};
  }
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {.resident_weights_bytes = 0,
            .state_capacity_bytes = 8 * 64,
            .per_request_state_bytes = 64,
            .temporary_scratch_bytes = 0,
            .retained_snapshot_capacity_bytes = std::size_t{1} << 25,
            .requires_device_runtime_lock = true};
  }
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= options_.width; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }
  [[nodiscard]] std::shared_ptr<const gufo::sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    if (!options_.constraint)
      return TextModelRunner::BuildConstraintVocabulary();
    return std::make_shared<const gufo::sampling::ConstraintVocabulary>(
        static_cast<std::uint32_t>(kVocab), [](std::uint32_t id) {
          return gufo::sampling::ConstraintVocabulary::Piece{
              .text = Piece(static_cast<TextRunnerToken>(id))};
        });
  }
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return {static_cast<TextRunnerToken>(text.size())};
  }
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }
  [[nodiscard]] std::optional<std::string> RenderPrompt(
      const ChatRequest&) const override {
    return "toy rendered prompt";
  }
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const auto token : tokens) {
      text += Piece(token);
    }
    return text;
  }
  static std::string Piece(TextRunnerToken token) {
    return " t" + std::to_string(token) + ";";
  }
  [[nodiscard]] std::unique_ptr<gufo::server::ComponentCacheResources>
  CreateComponentCacheResources(gufo::cache::ResourceLedger& ledger,
                                std::size_t) const override {
    return gufo::server::testing::MakeFakeCacheResources(
        ledger,
        std::make_unique<ToyAdapter>(options_, [this] { Record("reset"); }),
        {.private_blocks = 512,
         .row_backing_bytes = 1024,
         .total_row_bytes = options_.cache_row_bytes});
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      gufo::cache::Adapter& adapter, gufo::cache::Slot& slot) const override {
    std::size_t id = 0;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      id = slot_ids_.try_emplace(&slot, slot_ids_.size()).first->second;
    }
    auto state = std::make_unique<ToyState>(
        this, dynamic_cast<ToyAdapter&>(adapter), slot, id);
    Recall(*state);
    return state;
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t) const override {
    Recall(Toy(state));
  }
  /// Like Flash-Next's vision prompt: a different context resets a state that
  /// holds a sequence.
  void SetPromptContext(TextRunnerState& state,
                        std::shared_ptr<const gufo::server::TextPromptContext>
                            context) const override {
    const auto* toy = dynamic_cast<const ToyContext*>(context.get());
    if (context != nullptr && toy == nullptr) {
      throw std::invalid_argument("toy prompt context expected");
    }
    auto& target = Toy(state);
    const std::uint8_t shade = toy != nullptr ? toy->shade : 0;
    if (shade == 0 && target.shade == 0) {
      return;
    }
    if (!target.tokens.empty() && shade != target.shade) {
      (void)target.adapter.Invalidate(target.slot);
      target.tokens.clear();
    }
    target.shade = shade;
    // Rank 0's cache binds a fresh state where rank 1 clears its own, so only
    // a context that reaches the model is a call both ranks must make.
    if (shade != 0) {
      Record("context s" + std::to_string(target.id()) + " " +
             std::to_string(shade));
    }
  }
  [[nodiscard]] std::vector<std::uint8_t> EncodePromptContext(
      const gufo::server::TextPromptContext& context) const override {
    return {dynamic_cast<const ToyContext&>(context).shade};
  }
  [[nodiscard]] std::shared_ptr<const gufo::server::TextPromptContext>
  DecodePromptContext(std::span<const std::uint8_t> bytes) const override {
    if (options_.fail_context_decode || bytes.size() != 1) {
      throw std::invalid_argument("injected prompt context failure");
    }
    return std::make_shared<ToyContext>(bytes[0]);
  }
  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    Require(std::ranges::equal(Toy(state).tokens, prefix),
            "reused prefix equals state");
    Record("reuse " + std::to_string(prefix.size()));
  }
  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    if (options_.prefill_hook)
      options_.prefill_hook();
    auto& toy = Toy(state);
    if (offset != toy.tokens.size() || offset >= prompt.size()) {
      throw std::logic_error("toy prefill offset mismatch");
    }
    const auto consumed = std::min(
        {max_input_tokens, prompt.size() - offset, options_.prefill_chunk});
    Append(toy, prompt.subspan(offset, consumed));
    ++prefill_calls_;
    Record("prefill s" + std::to_string(toy.id()) + " @" +
           std::to_string(offset) + " +" + std::to_string(consumed) + "/" +
           std::to_string(prompt.size()));
    return {.consumed_tokens = consumed,
            .decode_ready = toy.tokens.size() == prompt.size()};
  }
  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state,
      gufo::sampling::SamplerState& sampler) const override {
    auto& toy = Toy(state);
    const auto logits = Logits(toy.tokens, toy.shade);
    const auto token = static_cast<TextRunnerToken>(sampler.Sample(logits));
    if (token == kEos) {
      return {.stop = true};
    }
    return {.stop = false, .token = token, .piece = Piece(token)};
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& toy = Toy(state);
    Append(toy, std::span(&token, 1));
    Record("advance s" + std::to_string(toy.id()) + " " +
           std::to_string(token));
  }
  void AdvanceBatch(std::span<const gufo::server::TextRunnerAdvance> advances)
      const override {
    if (advances.size() > 1) {
      Record("batch " + std::to_string(advances.size()));
    }
    TextModelRunner::AdvanceBatch(advances);
  }
  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      gufo::sampling::SamplerState& sampler) const override {
    if (!options_.mtp || max_tokens < 2) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    return Cycle(state, max_tokens, sampler, 3);
  }
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const gufo::server::TextRunnerDecode> decodes) const override {
    return DecodeBatchPlanned(decodes, PlanDecodeBatch(decodes));
  }
  [[nodiscard]] std::optional<std::uint32_t> PlanDecodeBatch(
      std::span<const gufo::server::TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || !options_.mtp ||
        std::ranges::any_of(decodes, [](const auto& decode) {
          return decode.sampler.get().config().uses_random_sampling();
        })) {
      return std::nullopt;
    }
    return options_.batch_plan;
  }
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatchPlanned(
      std::span<const gufo::server::TextRunnerDecode> decodes,
      std::optional<std::uint32_t> plan) const override {
    if (decodes.size() < 2 || !options_.mtp) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    Record("decode batch " + std::to_string(decodes.size()) + " plan " +
           (plan.has_value() ? std::to_string(*plan) : std::string("own")));
    std::vector<TextDecodeStep> steps;
    for (const auto& decode : decodes) {
      auto step = Cycle(decode.state.get(), decode.max_tokens,
                        decode.sampler.get(), plan.has_value() ? *plan + 1 : 3);
      step.execution_plan = {.kind = TextExecutionPlanKind::kBatched,
                             .physical_width = decodes.size()};
      steps.push_back(std::move(step));
    }
    return steps;
  }
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return dynamic_cast<const ToyState&>(state).tokens.size();
  }

private:
  /// One multi-token cycle of at most `limit` tokens.
  TextDecodeStep Cycle(TextRunnerState& state, std::size_t max_tokens,
                       gufo::sampling::SamplerState& sampler,
                       std::size_t limit) const {
    auto& toy = Toy(state);
    TextDecodeStep step;
    std::string produced;
    // Like Flash-Next: draw on a working copy, leave a deferred draw for the
    // next call when sampling at random, and publish the draw state back.
    gufo::sampling::SamplerState working = sampler;
    for (std::size_t i = 0; i < std::min(max_tokens, limit); ++i) {
      const auto token = static_cast<TextRunnerToken>(
          working.Sample(Logits(toy.tokens, toy.shade)));
      if (token == kEos) {
        step.stop = true;
        break;
      }
      Append(toy, std::span(&token, 1));
      step.selections.push_back({.token = token, .piece = Piece(token)});
      produced += " " + std::to_string(token);
    }
    if (!step.stop && sampler.config().uses_random_sampling()) {
      working.DeferSample(working.Sample(Logits(toy.tokens, toy.shade)));
      produced += " deferred";
    }
    sampler.CopyDrawStateFrom(working);
    step.draft_tokens = limit;
    step.draft_accepted_tokens =
        step.selections.empty() ? 0 : step.selections.size() - 1;
    if (options_.extra_draft_once) {
      options_.extra_draft_once = false;
      ++step.draft_tokens;
    }
    Record("decode s" + std::to_string(toy.id()) + " max" +
           std::to_string(max_tokens) + produced);
    return step;
  }
  static ToyState& Toy(TextRunnerState& state) {
    return dynamic_cast<ToyState&>(state);
  }
  /// Advances the sequence and the slot's rows together.
  static void Append(ToyState& toy, std::span<const TextRunnerToken> rows) {
    toy.adapter.Append(toy.slot, rows, rows);
    toy.tokens.insert(toy.tokens.end(), rows.begin(), rows.end());
    const std::lock_guard<std::mutex> lock(FrontierMutex());
    Frontiers()[toy.adapter.RecurrentHash(toy.slot)] = toy.tokens;
  }
  /// The slot's state decides the sequence, as restored rows do for a model.
  static void Recall(ToyState& toy) {
    const std::lock_guard<std::mutex> lock(FrontierMutex());
    const auto found = Frontiers().find(toy.adapter.RecurrentHash(toy.slot));
    toy.tokens = found == Frontiers().end() ? std::vector<TextRunnerToken>{}
                                            : found->second;
  }
  /// Shared across runners: a restart must recall what the disk restores.
  static std::mutex& FrontierMutex() {
    static std::mutex mutex;
    return mutex;
  }
  static std::unordered_map<std::uint64_t, std::vector<TextRunnerToken>>&
  Frontiers() {
    static std::unordered_map<std::uint64_t, std::vector<TextRunnerToken>> map;
    return map;
  }
  std::vector<float> Logits(const std::vector<TextRunnerToken>& tokens,
                            std::uint8_t shade) const {
    std::size_t sum = std::size_t{shade} * 5;
    for (const auto token : tokens) {
      sum += token;
    }
    // Never EOS unless the sequence is long, so budgets decide most lengths.
    std::size_t peak = (sum * 7 + tokens.size() * 3) % (kVocab - 1) + 1;
    if (tokens.size() >= 40) {
      peak = kEos;
    }
    if (options_.diverge_at == tokens.size()) {
      options_.diverge_at.reset();
      peak = peak % (kVocab - 1) + 1;
    }
    std::vector<float> logits(kVocab);
    for (std::size_t i = 0; i < kVocab; ++i) {
      logits[i] = -0.25F * static_cast<float>(i > peak ? i - peak : peak - i);
    }
    return logits;
  }

  mutable ToyOptions options_;
  mutable std::mutex mutex_;
  mutable CallLog log_;
  mutable std::unordered_map<const gufo::cache::Slot*, std::size_t> slot_ids_;
  mutable std::atomic<std::size_t> prefill_calls_{0};
  mutable std::atomic<std::size_t> cancellation_checks_{0};
};

void ToyState::Invalidate() noexcept {
  (void)adapter.Invalidate(slot);
  tokens.clear();
}

void ToyState::SetCancellationCheck(const CancellationCheck&) {
  owner_->CountCancellationCheck();
}

/// Rank 0 with the real scheduler, rank 1 with an executor, and the channel
/// between them. Both ranks hold `sessions` states.
class Pair {
public:
  /// With `disk`, rank 0's cache persists there, for both ranks' state.
  Pair(ToyOptions rank0_options, ToyOptions rank1_options,
       std::size_t sessions = 1, std::filesystem::path disk = {})
      : inner0_(std::make_shared<ToyRunner>(rank0_options)),
        runner1_(std::make_shared<ToyRunner>(rank1_options)) {
    const auto port = FreePort();
    std::string server_error;
    std::string client_error;
    std::thread connector([&] {
      client_ = TpControlChannel::Connect("127.0.0.1", port, &client_error);
    });
    server_ = TpControlChannel::Listen(port, &server_error);
    connector.join();
    Require(server_ != nullptr && client_ != nullptr,
            "channel pair: " + server_error + client_error);
    const TpControlConfig rank0{
        .rank = 0,
        .world_size = 2,
        .max_context = 256,
        .auth_token = "executor-test",
        .sessions = static_cast<std::uint32_t>(sessions)};
    TpControlConfig rank1 = rank0;
    rank1.rank = 1;
    bool server_ok = false;
    bool client_ok = false;
    std::thread handshake(
        [&] { client_ok = client_->Handshake(rank1, &client_error); });
    server_ok = server_->Handshake(rank0, &server_error);
    handshake.join();
    Require(server_ok && client_ok,
            "handshake: " + server_error + client_error);

    broker_ =
        std::make_shared<gufo::server::TpResponseBroker>(server_, sessions + 8);
    sink_ = std::make_shared<TpControlInstructionSink>(server_, broker_);
    std::shared_ptr<TpBulkSocket> bulk0, bulk1;
    std::string bulk_error;
    Require(TpBulkSocket::Pair(&bulk0, &bulk1, &bulk_error), bulk_error);
    // Rank 1 says hello on the bulk connection, which the mirrored runner
    // reads.
    executor_ = std::make_unique<TpExecutor>(runner1_, sessions, bulk1);
    mirrored_ = std::make_shared<TpMirroredRunner>(inner0_, sink_, bulk0);
    std::optional<gufo::server::TextRunnerDiskCacheOptions> disk_options;
    if (!disk.empty()) {
      disk_options = gufo::server::TextRunnerDiskCacheOptions{
          .directory = disk, .capacity_bytes = 1U << 22};
    }
    pool_ = std::make_shared<TextRunnerPool>(mirrored_, sessions,
                                             std::move(disk_options));
    scheduler_ = std::make_shared<TextGenerationScheduler>(pool_);
    worker_ = std::thread([this] { Work(); });
  }

  ~Pair() {
    scheduler_.reset();
    pool_.reset();
    broker_->FailAll("test finished");
    server_.reset();
    client_->Interrupt();
    worker_.join();
  }

  struct Outcome {
    TextGenerationBackend::Result result;
    std::string worker_error;
  };

  /// A request that finished on rank 0 but has not ended on rank 1 yet.
  struct Open {
    std::uint64_t sequence{0};
    Outcome outcome;
  };

  /// Runs one request as serving does. `submitted` runs once the scheduler
  /// has the request.
  Outcome Run(
      std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
      gufo::sampling::SamplingConfig sampling = {},
      std::vector<std::string> stop_sequences = {},
      TextGenerationScheduler::CancellationCheck is_cancelled = {},
      bool reuse = false, std::size_t prefix = 0,
      const std::function<void()>& submitted = {},
      std::shared_ptr<const gufo::server::TextPromptContext> images = {}) {
    return Close(RunOpen(std::move(prompt), max_tokens, std::move(sampling),
                         std::move(stop_sequences), std::move(is_cancelled),
                         reuse, prefix, submitted, std::move(images)));
  }

  /// Runs a request on rank 0 without ending it, as when the next request
  /// arrives before rank 1's verdict on this one.
  Open RunOpen(
      std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
      gufo::sampling::SamplingConfig sampling = {},
      std::vector<std::string> stop_sequences = {},
      TextGenerationScheduler::CancellationCheck is_cancelled = {},
      bool reuse = false, std::size_t prefix = 0,
      const std::function<void()>& submitted = {},
      std::shared_ptr<const gufo::server::TextPromptContext> images = {}) {
    const auto sequence = next_sequence_.fetch_add(1);
    TpControlCommand begin{.sequence = sequence,
                           .max_tokens = static_cast<std::uint32_t>(max_tokens),
                           .client_id = "executor-test",
                           .sampling = sampling};
    if (images != nullptr) {
      begin.prompt_context = mirrored_->EncodePromptContext(*images);
    }
    for (const auto token : prompt) {
      begin.prompt_tokens.push_back(static_cast<std::int32_t>(token));
    }
    std::string error;
    Require(broker_->RegisterPendingResponse(sequence, &error),
            "register: " + error);
    Require(server_->SendCommand(begin, &error), "begin: " + error);
    mirrored_->BeginRequest(sequence);
    TextRequestMetadata metadata;
    metadata.prompt_context =
        std::make_shared<TpRequestContext>(sequence, std::move(images));
    metadata.cache_prompt = reuse;
    metadata.cache_prefix_tokens = prefix;
    metadata.stop_sequences = std::move(stop_sequences);
    auto request = scheduler_->Submit(std::move(prompt), max_tokens, sampling,
                                      is_cancelled, reuse, std::move(metadata));
    if (submitted) {
      submitted();
    }
    Open open{.sequence = sequence};
    try {
      open.outcome.result = request.Wait();
    } catch (const std::exception& exception) {
      open.outcome.worker_error = exception.what();
    }
    return open;
  }

  /// Ends a request on rank 1 and settles it as serving does.
  Outcome Close(Open open) {
    const auto sequence = open.sequence;
    Outcome outcome = std::move(open.outcome);
    const std::string local_error = std::move(outcome.worker_error);
    std::string error;
    Require(mirrored_->EndRequest(sequence, &error), "end: " + error);
    TpControlResponse response;
    Require(broker_->WaitForResponse(sequence, &response, &error),
            "response: " + error);
    Require(response.sequence == sequence, "response sequence");
    const bool dependencies = mirrored_->AwaitDependencies(sequence);
    mirrored_->Settle(sequence, response.error.empty() && dependencies);
    outcome.worker_error = !response.error.empty() ? response.error
                           : !dependencies
                               ? std::string("reused a rejected state")
                               : local_error;
    return outcome;
  }

  /// A reset sent between requests, as the component cache may send one.
  void ResetBetweenRequests() {
    std::string error;
    Require(sink_->Send(0, {.op = TpInstructionOp::kInvalidate, .state = 0},
                        nullptr, &error),
            "idle reset: " + error);
  }

  void RequireSameCalls(const std::string& what) const {
    const auto rank0 = inner0_->Log();
    const auto rank1 = runner1_->Log();
    if (rank0 != rank1) {
      std::fprintf(stderr, "rank 0 calls:\n");
      for (const auto& line : rank0) {
        std::fprintf(stderr, "  %s\n", line.c_str());
      }
      std::fprintf(stderr, "rank 1 calls:\n");
      for (const auto& line : rank1) {
        std::fprintf(stderr, "  %s\n", line.c_str());
      }
    }
    Require(rank0 == rank1, what + ": both ranks make the same model calls");
    Require(worker_failure_.empty(), what + ": worker " + worker_failure_);
  }

  std::size_t open_requests() const { return worker_open_.load(); }
  /// Ends a request on both ranks but never records rank 1's verdict.
  void EndWithoutVerdict(const Open& open) {
    std::string error;
    Require(mirrored_->EndRequest(open.sequence, &error), "end: " + error);
    TpControlResponse response;
    Require(broker_->WaitForResponse(open.sequence, &response, &error),
            "response: " + error);
  }

  /// Fails the next restore on the named rank.
  void FailNextRestore(bool rank0) {
    (rank0 ? inner0_ : runner1_)->FailNextRestore();
  }
  void SetDependencyGrace(std::chrono::milliseconds grace) {
    mirrored_->SetDependencyGrace(grace);
  }

  const ToyRunner& rank0() const { return *inner0_; }
  const ToyRunner& rank1() const { return *runner1_; }

private:
  void Work() {
    const auto respond = [this](const TpControlResponse& response,
                                std::string* error) {
      // Published before a verdict leaves, so a test that has its verdict
      // sees the worker's tables as they were then.
      worker_open_ = executor_->open_requests();
      return client_->SendResponse(response, error);
    };
    for (;;) {
      TpControlCommand command;
      std::string error;
      if (!client_->ReceiveCommand(&command, &error)) {
        return;
      }
      if (!executor_->Execute(command, respond, &error)) {
        worker_failure_ = error;
        return;
      }
    }
  }

  std::shared_ptr<ToyRunner> inner0_;
  std::shared_ptr<ToyRunner> runner1_;
  std::shared_ptr<TpControlChannel> server_;
  std::shared_ptr<TpControlChannel> client_;
  std::shared_ptr<gufo::server::TpResponseBroker> broker_;
  std::shared_ptr<TextRunnerPool> pool_;
  std::shared_ptr<TpControlInstructionSink> sink_;
  std::shared_ptr<TpMirroredRunner> mirrored_;
  std::shared_ptr<TextGenerationScheduler> scheduler_;
  std::unique_ptr<TpExecutor> executor_;
  std::thread worker_;
  std::string worker_failure_;
  std::atomic<std::size_t> worker_open_{0};
  std::atomic<std::uint64_t> next_sequence_{1};
};

/// Records rank 0's instructions and runs cache calls at once, for tests of
/// the mirrored runner alone.
class CaptureSink final : public TpInstructionSink {
public:
  struct Sent {
    std::uint64_t sequence;
    TpInstruction instruction;
  };
  void Synchronize(std::uint64_t sequence, TpInstruction instruction,
                   const std::function<void()>& local) override {
    instruction.index = next_++;
    sent.push_back({sequence, std::move(instruction)});
    local();
  }
  bool Send(std::uint64_t sequence, TpInstruction instruction,
            std::uint64_t* index, std::string*) override {
    instruction.index = next_;
    if (index != nullptr) {
      *index = next_;
    }
    ++next_;
    sent.push_back({sequence, std::move(instruction)});
    return true;
  }
  std::vector<Sent> sent;

private:
  std::uint64_t next_{0};
};

/// Records which instruction indices had a collective scope bound.
class RecordingScope final : public TpCallScope {
public:
  bool Begin(std::uint64_t index, std::string*) override {
    Require(!bound_.has_value(), "one collective scope at a time");
    bound_ = index;
    began.push_back(index);
    return true;
  }
  bool End(std::uint64_t index, std::string*) override {
    Require(bound_ == index, "a scope ends as it began");
    bound_.reset();
    return true;
  }
  std::vector<std::uint64_t> began;

private:
  std::optional<std::uint64_t> bound_;
};

std::size_t Count(const CallLog& log, const std::string& prefix) {
  return static_cast<std::size_t>(std::count_if(
      log.begin(), log.end(),
      [&](const std::string& line) { return line.rfind(prefix, 0) == 0; }));
}

/// Holds the scheduler in its first prefill until `count` requests were
/// submitted, so they run concurrently however the threads are scheduled.
class SubmissionGate {
public:
  explicit SubmissionGate(std::size_t count) : count_(count) {}
  void Submitted() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      ++submitted_;
    }
    condition_.notify_all();
  }
  std::function<void()> Hook() {
    return [this] {
      if (passed_.exchange(true)) {
        return;
      }
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this] { return submitted_ >= count_; });
    };
  }

private:
  const std::size_t count_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::size_t submitted_{0};
  std::atomic<bool> passed_{false};
};

/// Every capability by name. A capability added to TextRunnerCapabilities
/// stops this from compiling until TP2 decides how to carry it: forward it
/// and mirror on rank 1 every call it enables, or list it in
/// kTp2CapabilityGaps. Give it its capable value in the check in main too.
std::vector<std::pair<std::string_view, std::size_t>> CapabilityFields(
    const TextRunnerCapabilities& capabilities) {
  const auto& [supports_vision, incremental_prefill,
               final_token_advance_required, incremental_text_is_exact,
               multi_token_decode, batched_multi_token_decode,
               batched_multi_token_decode_max_width, prefix_reuse,
               in_pass_checkpoint, partial_prefill_retention, prefill_rollback,
               prefill_alignment, max_in_pass_checkpoints] = capabilities;
  return {{"supports_vision", supports_vision},
          {"incremental_prefill", incremental_prefill},
          {"final_token_advance_required", final_token_advance_required},
          {"incremental_text_is_exact", incremental_text_is_exact},
          {"multi_token_decode", multi_token_decode},
          {"batched_multi_token_decode", batched_multi_token_decode},
          {"batched_multi_token_decode_max_width",
           batched_multi_token_decode_max_width},
          {"prefix_reuse", prefix_reuse},
          {"in_pass_checkpoint", in_pass_checkpoint},
          {"partial_prefill_retention", partial_prefill_retention},
          {"prefill_rollback", prefill_rollback},
          {"prefill_alignment", prefill_alignment},
          {"max_in_pass_checkpoints", max_in_pass_checkpoints}};
}

/// Capabilities TP2 turns off because rank 1 cannot mirror them yet. Each is
/// a way a TP2 pair falls behind one host; closing one removes it here.
constexpr std::array<std::string_view, 2> kTp2CapabilityGaps{
    "in_pass_checkpoint",  // PrefillThrough: an in-pass prompt checkpoint
    "prefill_rollback",    // a prefill interrupted mid-pass and rolled back
};

/// A mirrored runner alone, for tests of what it sends: rank 1's executor only
/// says hello. Its states come from the cache resources, as the pool's do.
class Direct {
public:
  explicit Direct(
      std::shared_ptr<ToyRunner> inner,
      std::shared_ptr<TpInstructionSink> sink = std::make_shared<CaptureSink>(),
      std::shared_ptr<TpCallScope> scope = {}) {
    std::shared_ptr<TpBulkSocket> bulk0, bulk1;
    std::string error;
    Require(TpBulkSocket::Pair(&bulk0, &bulk1, &error), error);
    executor_ = std::make_unique<TpExecutor>(
        std::make_shared<ToyRunner>(ToyOptions{}), 1, bulk1);
    runner = std::make_shared<TpMirroredRunner>(std::move(inner),
                                                std::move(sink), bulk0, scope);
  }
  std::unique_ptr<TextRunnerState> NewState() {
    if (resources_ == nullptr) {
      resources_ = runner->CreateComponentCacheResources(ledger_, 0);
    }
    slots_.push_back(resources_->adapter->CreateSlot(guard_));
    return runner->BindComponentState(*resources_->adapter, *slots_.back());
  }

private:
  class NoBorrowers final : public gufo::cache::MutationGuard {
    void BeforeOverwrite(gufo::cache::ComponentId, gufo::cache::Rows,
                         gufo::cache::Rows) override {}
    void BeforeRelease(gufo::cache::ComponentId, gufo::cache::Rows,
                       gufo::cache::Rows) noexcept override {}
  };
  std::unique_ptr<TpExecutor> executor_;

public:
  std::shared_ptr<TpMirroredRunner> runner;

private:
  gufo::cache::ResourceLedger ledger_{
      gufo::cache::ResourceLimits{std::size_t{1} << 40, std::size_t{1} << 40,
                                  std::size_t{1} << 40, std::size_t{1} << 40}};
  std::unique_ptr<gufo::server::ComponentCacheResources> resources_;
  NoBorrowers guard_;
  std::vector<std::unique_ptr<gufo::cache::Slot>> slots_;
};

/// The rank-1 end of a bulk connection no rank 0 reads, for an executor driven
/// by hand. The other end stays open for the whole test.
std::shared_ptr<TpBulkSocket> UnreadBulk() {
  static std::vector<std::shared_ptr<TpBulkSocket>> kept;
  std::shared_ptr<TpBulkSocket> bulk0, bulk1;
  std::string error;
  Require(TpBulkSocket::Pair(&bulk0, &bulk1, &error), error);
  kept.push_back(std::move(bulk0));
  return bulk1;
}

/// Waits until rank 0's cache has published a disk entry.
void WaitForDisk(const std::filesystem::path& directory) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < deadline) {
    std::error_code error;
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(directory, error)) {
      if (entry.is_regular_file(error) && entry.path().extension() == ".bin") {
        return;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  Require(false, "rank 0 persists the pair's checkpoint");
}

/// The bulk connection: frames either way, and only for the control token.
void TestBulkSocket() {
  {
    std::shared_ptr<TpBulkSocket> a, b;
    std::string error;
    Require(TpBulkSocket::Pair(&a, &b, &error), error);
    const std::array<std::byte, 3> payload{std::byte{1}, std::byte{2},
                                           std::byte{3}};
    Require(!a->Send({.kind = TpBulkKind::kOut, .bytes = 2}, payload, &error),
            "a frame's size must match its payload");
    Require(a->Send({.kind = TpBulkKind::kOut,
                     .state = 1,
                     .component = 2,
                     .is_private = 1,
                     .transfer = 3,
                     .first = 4,
                     .end = 5,
                     .bytes = payload.size()},
                    payload, &error),
            error);
    TpBulkHeader header;
    std::array<std::byte, 3> received{};
    Require(b->ReceiveHeader(&header, &error) &&
                b->ReceivePayload(received, &error),
            error);
    Require(header.kind == TpBulkKind::kOut && header.state == 1 &&
                header.component == 2 && header.is_private == 1 &&
                header.transfer == 3 && header.first == 4 && header.end == 5 &&
                header.bytes == 3 && received == payload,
            "a bulk frame arrives whole");
    a->Shutdown();
    Require(!b->ReceiveHeader(&header, &error), "a lost peer ends the wait");
  }
  for (const bool same_token : {true, false}) {
    const auto port = FreePort();
    std::shared_ptr<TpBulkSocket> rank1;
    std::string connect_error;
    std::thread connector([&] {
      rank1 = TpBulkSocket::Connect(
          "127.0.0.1", port, same_token ? "token" : "other", &connect_error);
    });
    std::string listen_error;
    const auto rank0 = TpBulkSocket::Listen(port, "token", &listen_error);
    connector.join();
    Require(
        (rank0 != nullptr) == same_token && (rank1 != nullptr) == same_token,
        "the bulk connection proves the control token: " + listen_error +
            " / " + connect_error);
  }
}

gufo::sampling::SamplingConfig Sampled() {
  gufo::sampling::SamplingConfig config;
  config.temperature = 0.9F;
  config.seed = 42;
  // History matters only with a penalty, so this also checks that rank 1's
  // sampler accepts the same tokens as rank 0's.
  config.repeat_penalty = 1.3F;
  return config;
}

}  // namespace

int main() {
  TestBulkSocket();
  {
    auto toy = std::make_shared<ToyRunner>(ToyOptions{});
    Direct direct(toy);
    const auto& runner = *direct.runner;
    Require(runner.RenderPrompt(ChatRequest{}) == "toy rendered prompt",
            "TP runner forwards rendered prompts");
    using Status = TextModelRunner::DeviceProbeStatus;
    for (const auto status :
         {Status::kUsable, Status::kPending, Status::kLost}) {
      toy->probe_status = status;
      Require(runner.PollDevice() == status &&
                  runner.DeviceUsable() == (status != Status::kLost),
              "TP scheduler observes the underlying GPU probe result");
    }
  }
  const std::vector<TextRunnerToken> prompt{5, 9, 13, 17, 21, 3, 8};

  // TP2 keeps up with one host: the wrapper reports every capability of the
  // runner it wraps, apart from the listed gaps.
  {
    const TextRunnerCapabilities every{
        .supports_vision = true,
        .incremental_prefill = true,
        .final_token_advance_required = false,
        .incremental_text_is_exact = true,
        .multi_token_decode = true,
        .batched_multi_token_decode = true,
        .batched_multi_token_decode_max_width = 8,
        .prefix_reuse = true,
        .in_pass_checkpoint = true,
        .partial_prefill_retention = true,
        .prefill_rollback = true,
        .prefill_alignment = 4,
        .max_in_pass_checkpoints = 3,
    };
    auto toy = std::make_shared<ToyRunner>(ToyOptions{.capabilities = every});
    Direct direct(toy);
    const auto& runner = *direct.runner;
    const auto inner = CapabilityFields(toy->Descriptor().capabilities);
    const auto wrapped = CapabilityFields(runner.Descriptor().capabilities);
    for (std::size_t i = 0; i < inner.size(); ++i) {
      const std::string name(inner[i].first);
      if (std::ranges::find(kTp2CapabilityGaps, inner[i].first) !=
          kTp2CapabilityGaps.end()) {
        Require(
            wrapped[i].second != inner[i].second,
            "TP2 now carries " + name + ": remove it from kTp2CapabilityGaps");
      } else {
        Require(wrapped[i].second == inner[i].second,
                "TP2 drops " + name +
                    ": mirror it on rank 1 or list it in kTp2CapabilityGaps");
      }
    }
  }

  for (const bool mtp : {false, true}) {
    Pair pair({.mtp = mtp, .cache = true}, {.mtp = mtp, .cache = true});
    const auto sampling = mtp ? Sampled() : gufo::sampling::SamplingConfig{};
    const auto first = pair.Run(prompt, 6, sampling, {}, {}, true);
    Require(first.worker_error.empty(), "cached first: " + first.worker_error);
    const auto again = pair.Run(prompt, 6, sampling, {}, {}, true);
    Require(again.worker_error.empty(),
            "cached restore: " + again.worker_error);
    Require(first.result.tokens == again.result.tokens,
            "restore preserves outputs");
    Require(again.result.cached_prompt_tokens == prompt.size(),
            "identical replay restores the whole prompt, as on one host");
    pair.RequireSameCalls("checkpoint reuse");
    auto branch = prompt;
    branch.push_back(11);
    const auto fork = pair.Run(branch, 4, sampling, {}, {}, true);
    Require(fork.worker_error.empty(), "branch: " + fork.worker_error);
    Require(fork.result.cached_prompt_tokens == prompt.size(),
            "older boundary restored");
    pair.RequireSameCalls("branch reuse");
    (void)pair.Run(prompt, 6, sampling, {}, {}, true);
    auto continuation = prompt;
    continuation.insert(continuation.end(), again.result.tokens.begin(),
                        again.result.tokens.end());
    continuation.push_back(7);
    const auto live = pair.Run(continuation, 4, sampling, {}, {}, true);
    Require(live.worker_error.empty(), "live: " + live.worker_error);
    Require(live.result.cached_prompt_tokens > prompt.size(),
            "live frontier reused");
    pair.RequireSameCalls("live reuse");
  }
  // A prompt not on the model's alignment captures the aligned boundary before
  // it: the pool sizes that capture before prefilling up to it.
  {
    ToyOptions aligned{
        .cache = true, .prefill_chunk = 4, .prefill_alignment = 4};
    Pair pair(aligned, aligned);
    const auto first = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(first.worker_error.empty(), "aligned: " + first.worker_error);
    auto longer = prompt;
    longer.push_back(11);
    const auto reused = pair.Run(longer, 3, {}, {}, {}, true);
    Require(
        reused.worker_error.empty() && reused.result.cached_prompt_tokens == 4,
        "a longer prompt reuses the aligned boundary: " + reused.worker_error);
    pair.RequireSameCalls("aligned reuse");
  }
  // A capture that fails on either rank only skips the checkpoint, as on one
  // host: the request succeeds, and outputs do not change.
  for (const bool rank0_fails : {false, true}) {
    Pair pair({.cache = true, .fail_capture_once = rank0_fails},
              {.cache = true, .fail_capture_once = !rank0_fails});
    const auto first = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(first.worker_error.empty(),
            "failed capture is not a failed request: " + first.worker_error);
    const auto next = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(next.worker_error.empty(),
            "request after failed capture: " + next.worker_error);
    Require(next.result.tokens == first.result.tokens,
            "a skipped checkpoint keeps outputs");
    const auto reused = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(reused.worker_error.empty() &&
                reused.result.cached_prompt_tokens == prompt.size() &&
                reused.result.tokens == first.result.tokens,
            "the next capture succeeds on both ranks");
  }
  // A restore that fails on either rank is a miss: the request recomputes and
  // the checkpoint an agreed request captured stays reusable.
  for (const bool rank0_fails : {false, true}) {
    Pair pair({.cache = true}, {.cache = true});
    const auto first = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(first.worker_error.empty(), first.worker_error);
    pair.FailNextRestore(rank0_fails);
    const auto failed = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(failed.worker_error.empty() &&
                failed.result.tokens == first.result.tokens,
            "a failed restore recomputes: " + failed.worker_error);
    const auto recovered = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(recovered.worker_error.empty() &&
                recovered.result.cached_prompt_tokens == prompt.size() &&
                recovered.result.tokens == first.result.tokens,
            "the next request reuses again: " + recovered.worker_error);
  }

  // Eviction involves only rank 0's table; what is left still restores.
  {
    Pair pair({.cache = true, .cache_row_bytes = 8192},
              {.cache = true, .cache_row_bytes = 8192});
    std::vector<std::vector<TextRunnerToken>> outputs;
    for (int i = 0; i < 4; ++i) {
      auto distinct = prompt;
      distinct[0] += i;
      const auto run = pair.Run(distinct, 3, {}, {}, {}, true);
      Require(run.worker_error.empty(), "eviction: " + run.worker_error);
      outputs.push_back(run.result.tokens);
    }
    for (int i = 0; i < 4; ++i) {
      auto distinct = prompt;
      distinct[0] += i;
      const auto run = pair.Run(distinct, 3, {}, {}, {}, true);
      Require(run.worker_error.empty() && run.result.tokens == outputs[i],
              "a request after eviction keeps its output: " + run.worker_error);
    }
    pair.RequireSameCalls("eviction");
  }
  {
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    std::atomic<bool> cancelled{false};
    ToyOptions held{.cache = true};
    std::once_flag held_once;
    held.capture_hook = [&] {
      std::call_once(held_once, [&] { entered.set_value(); });
      released.wait();
    };
    Pair pair(held, {.cache = true});
    auto pending = std::async(std::launch::async, [&] {
      return pair.Run(
          prompt, 6, {}, {}, [&] { return cancelled.load(); }, true);
    });
    Require(entered.get_future().wait_for(std::chrono::seconds(5)) ==
                std::future_status::ready,
            "asynchronous capture starts");
    Require(Count(pair.rank0().Log(), "advance") == 0,
            "capture freezes the state before advance");
    cancelled = true;
    release.set_value();
    const auto stopped = pending.get();
    Require(stopped.worker_error.empty(),
            "cancel during capture: " + stopped.worker_error);
    const auto next = pair.Run(prompt, 3, {}, {}, {}, true);
    Require(next.worker_error.empty() && next.result.cached_prompt_tokens > 0,
            "joined capture survives cancellation");
    pair.RequireSameCalls("capture cancellation reuse");
  }

  // Mutating the instruction stream must fail even though every surviving
  // frame is valid on its own and has a freshly contiguous transport index.
  for (int mutation = 0; mutation < 4; ++mutation) {
    auto runner = std::make_shared<ToyRunner>(ToyOptions{.cache = true});
    TpExecutor executor(runner, 1, UnreadBulk());
    const TpControlCommand begin{
        .sequence = 1, .max_tokens = 1, .prompt_tokens = {5, 9}};
    std::vector<gufo::server::TpInstruction> calls{
        {.op = TpInstructionOp::kPrefill, .count = 2, .prompt_size = 2},
        {.op = TpInstructionOp::kReuse, .prompt_size = 2},
        {.op = TpInstructionOp::kCancelPrepare}};
    gufo::server::TpExecutionDigest digest;
    for (const auto& call : calls) {
      const std::vector<TextRunnerToken> prefix{5, 9};
      gufo::server::DigestTpCall(digest, call, prefix);
      if (call.op == TpInstructionOp::kPrefill)
        gufo::server::DigestTpPrefill(
            digest, {.consumed_tokens = 2, .decode_ready = true}, 2);
      if (call.op == TpInstructionOp::kReuse)
        digest.Add(2);
    }
    if (mutation == 1)
      calls.pop_back();  // missing call
    if (mutation == 2)
      calls.erase(calls.begin() + 1);  // missing reuse
    if (mutation == 3)
      std::swap(calls[1], calls[2]);  // reordered calls
    calls.push_back(
        {.op = TpInstructionOp::kEnd, .count = 3, .digest = digest.value()});
    std::optional<std::string> outcome;
    const auto respond = [&](const TpControlResponse& response, std::string*) {
      if (response.kind == gufo::server::TpControlResponseKind::kSingle) {
        outcome = response.error;
      }
      return true;
    };
    std::string error;
    Require(executor.Execute(begin, respond, &error), error);
    for (std::size_t next = 0; next < calls.size(); ++next) {
      calls[next].index = next;
      Require(executor.Execute({.sequence = 1,
                                .kind = TpControlCommandKind::kInstruction,
                                .instruction = calls[next]},
                               respond, &error),
              error);
    }
    Require(outcome.has_value() && outcome->empty() == (mutation == 0),
            "instruction mutation is detected: " + outcome.value_or("none"));
    Require(executor.open_requests() == 0, "the verdict closes the request");
  }

  // Rank 1 refuses a stream that names requests it does not know.
  {
    auto runner = std::make_shared<ToyRunner>(ToyOptions{});
    TpExecutor executor(runner, 2, UnreadBulk());
    const auto respond = [](const TpControlResponse&, std::string*) {
      return true;
    };
    std::string error;
    Require(!executor.Execute({.sequence = 3,
                               .kind = TpControlCommandKind::kInstruction,
                               .instruction = {.op = TpInstructionOp::kPrefill,
                                               .count = 1,
                                               .prompt_size = 1}},
                              respond, &error) &&
                error.find("not open") != std::string::npos,
            "an instruction for an unknown request is a protocol failure: " +
                error);
    TpExecutor batch_executor(runner, 2, UnreadBulk());
    Require(batch_executor.Execute(
                {.sequence = 1, .max_tokens = 1, .prompt_tokens = {5}}, respond,
                &error),
            error);
    Require(
        !batch_executor.Execute(
            {.sequence = 0,
             .kind = TpControlCommandKind::kInstruction,
             .instruction = {.op = TpInstructionOp::kAdvanceBatch,
                             .batch = {{.sequence = 1, .state = 0},
                                       {.sequence = 2, .state = 1}}}},
            respond, &error) &&
            error.find("not open") != std::string::npos,
        "a batch member of an unknown request is a protocol failure: " + error);
  }

  // A batch member whose rebuilt constraint refuses rank 0's token fails its
  // request; the batch and rank 1 go on.
  {
    auto runner =
        std::make_shared<ToyRunner>(ToyOptions{.width = 2, .constraint = true});
    TpExecutor executor(runner, 2, UnreadBulk());
    std::optional<std::string> verdict;
    const auto respond = [&](const TpControlResponse& response, std::string*) {
      if (response.kind == gufo::server::TpControlResponseKind::kSingle)
        verdict = response.error;
      return true;
    };
    std::string error;
    Require(executor.Execute(
                {.sequence = 1,
                 .max_tokens = 4,
                 .prompt_tokens = {5},
                 .constrained = true,
                 .constraint_source =
                     gufo::server::TpConstraintSource{
                         .response_format_json = R"({"type":"json_object"})"}},
                respond, &error),
            error);
    Require(
        executor.Execute({.sequence = 2, .max_tokens = 4, .prompt_tokens = {5}},
                         respond, &error),
        error);
    Require(executor.Execute(
                {.sequence = 0,
                 .kind = TpControlCommandKind::kInstruction,
                 .instruction =
                     {.op = TpInstructionOp::kAdvanceBatch,
                      .batch = {{.sequence = 1, .state = 0, .token = 7},
                                {.sequence = 2, .state = 1, .token = 7}}}},
                respond, &error),
            "a refused batch token is not a protocol failure: " + error);
    Require(executor.Execute({.sequence = 1,
                              .kind = TpControlCommandKind::kInstruction,
                              .instruction = {.op = TpInstructionOp::kEnd,
                                              .index = 1,
                                              .count = 1}},
                             respond, &error),
            error);
    Require(verdict.has_value() &&
                verdict->find("batched advance failed") != std::string::npos,
            "the refused member's verdict names the failure: " +
                verdict.value_or("none"));
    Require(executor.open_requests() == 1, "the other member stays open");
  }

  // The mirrored runner alone: every call belongs to the request its state is
  // leased to, forwards bind their instruction's collective scope, a batch is
  // one instruction, and a request that reuses what another computed depends
  // on rank 1's verdict on it.
  {
    auto toy = std::make_shared<ToyRunner>(ToyOptions{.cache = true});
    auto sink = std::make_shared<CaptureSink>();
    auto scope = std::make_shared<RecordingScope>();
    Direct direct(toy, sink, scope);
    auto runner = direct.runner;
    auto first = direct.NewState();
    auto second = direct.NewState();
    const std::vector<TextRunnerToken> short_prompt{4, 5, 6};
    bool refused = false;
    try {
      (void)runner->Prefill(*first, short_prompt, 0, 3);
    } catch (const std::logic_error&) {
      refused = true;
    }
    Require(refused && sink->sent.empty(),
            "a call on a state no request leases is refused unsent");

    runner->BeginRequest(7);
    runner->BeginRequest(8);
    runner->SetPromptContext(*first, std::make_shared<TpRequestContext>(7));
    runner->SetPromptContext(*second, std::make_shared<TpRequestContext>(8));
    (void)runner->Prefill(*first, short_prompt, 0, 3);
    (void)runner->Prefill(*second, short_prompt, 0, 3);
    // Leasing a state binds it on rank 1 too, before the request's calls.
    Require(
        sink->sent.size() == 4 &&
            sink->sent[0].instruction.op == TpInstructionOp::kPromptContext &&
            sink->sent[0].sequence == 7 &&
            sink->sent[1].instruction.op == TpInstructionOp::kPromptContext &&
            sink->sent[1].sequence == 8 && sink->sent[2].sequence == 7 &&
            sink->sent[3].sequence == 8,
        "each call names the request its state is leased to");

    std::exception_ptr failures[2];
    const gufo::server::TextRunnerAdvance advances[] = {
        {.state = *first, .token = 9, .failure = &failures[0]},
        {.state = *second, .token = 10, .failure = &failures[1]}};
    runner->AdvanceBatch(advances);
    const auto& batch = sink->sent.back();
    Require(batch.sequence == 0 &&
                batch.instruction.op == TpInstructionOp::kAdvanceBatch &&
                batch.instruction.batch.size() == 2 &&
                batch.instruction.batch[0].sequence == 7 &&
                batch.instruction.batch[0].state == 0 &&
                batch.instruction.batch[0].token == 9 &&
                batch.instruction.batch[1].sequence == 8 &&
                batch.instruction.batch[1].state == 1 && !failures[0] &&
                !failures[1],
            "a batched advance is one instruction naming each member");
    Require(Count(toy->Log(), "batch 2") == 1,
            "rank 0 runs the batch as one call");
    std::vector<std::uint64_t> forwards;
    for (const auto& sent : sink->sent) {
      if (sent.instruction.op == TpInstructionOp::kPrefill ||
          sent.instruction.op == TpInstructionOp::kAdvanceBatch) {
        forwards.push_back(sent.instruction.index);
      }
    }
    Require(scope->began == forwards,
            "every forward binds its instruction's collective scope");

    std::string error;
    Require(runner->EndRequest(7, &error) && runner->EndRequest(8, &error),
            error);
    Require(!runner->EndRequest(7, &error), "a request ends once");
    Require(runner->CanContinue(*first, 3) && runner->CanContinue(*second, 3),
            "an ended request's state is reusable before its verdict");
    // Requests that reuse those states depend on the verdicts.
    runner->BeginRequest(9);
    runner->BeginRequest(10);
    runner->SetPromptContext(*first, std::make_shared<TpRequestContext>(9));
    runner->SetPromptContext(*second, std::make_shared<TpRequestContext>(10));
    runner->Settle(7, true);
    runner->Settle(8, false);
    Require(runner->AwaitDependencies(9),
            "a request that reused an agreed state stands");
    Require(!runner->AwaitDependencies(10),
            "a request that reused a rejected state fails too");
    Require(runner->CanContinue(*first, 3),
            "an agreed request's state is reusable");
    Require(!runner->CanContinue(*second, 3),
            "a rejected request's state is not reused");
    second->Invalidate();
    Require(runner->CanContinue(*second, 3), "a reset state is clean");
    Require(runner->EndRequest(9, &error) && runner->EndRequest(10, &error),
            error);
    bool unleased = false;
    try {
      runner->Advance(*first, 5);
    } catch (const std::logic_error&) {
      unleased = true;
    }
    Require(unleased, "an ended request releases its state");
  }

  // A batched multi-token decode is one instruction carrying each member's
  // budget and draw state and rank 0's draft plan.
  {
    auto toy = std::make_shared<ToyRunner>(
        ToyOptions{.mtp = true, .width = 2, .batch_plan = 1});
    auto sink = std::make_shared<CaptureSink>();
    auto scope = std::make_shared<RecordingScope>();
    Direct direct(toy, sink, scope);
    auto runner = direct.runner;
    Require(runner->Descriptor().capabilities.batched_multi_token_decode,
            "TP2 offers batched multi-token decoding");
    auto first = direct.NewState();
    auto second = direct.NewState();
    const std::vector<TextRunnerToken> short_prompt{4, 5, 6};
    runner->BeginRequest(7);
    runner->BeginRequest(8);
    runner->SetPromptContext(*first, std::make_shared<TpRequestContext>(7));
    runner->SetPromptContext(*second, std::make_shared<TpRequestContext>(8));
    (void)runner->Prefill(*first, short_prompt, 0, 3);
    (void)runner->Prefill(*second, short_prompt, 0, 3);
    gufo::sampling::SamplerState samplers[] = {
        gufo::sampling::SamplerState{gufo::sampling::SamplingConfig{}},
        gufo::sampling::SamplerState{gufo::sampling::SamplingConfig{}}};
    const gufo::server::TextRunnerDecode decodes[] = {
        {.state = *first, .max_tokens = 5, .sampler = samplers[0]},
        {.state = *second, .max_tokens = 1, .sampler = samplers[1]}};
    const auto steps = runner->DecodeBatch(decodes);
    const auto& batch = sink->sent.back();
    Require(batch.sequence == 0 &&
                batch.instruction.op == TpInstructionOp::kDecodeBatch &&
                batch.instruction.batch_drafts == 2 &&
                batch.instruction.batch.size() == 2 &&
                batch.instruction.batch[0].sequence == 7 &&
                batch.instruction.batch[0].count == 5 &&
                batch.instruction.batch[0].pending == -1 &&
                batch.instruction.batch[1].sequence == 8 &&
                batch.instruction.batch[1].state == 1 &&
                batch.instruction.batch[1].count == 1,
            "a batched decode is one instruction naming each member");
    Require(steps.size() == 2 && steps[0].selections.size() == 2 &&
                steps[1].selections.size() == 1 && !steps[0].failure &&
                !steps[1].failure,
            "each member decodes within the plan and its budget");
    Require(Count(toy->Log(), "decode batch 2 plan 1") == 1,
            "rank 0 runs the batch as one call with its plan");
    Require(
        !scope->began.empty() && scope->began.back() == batch.instruction.index,
        "the batch binds its instruction's collective scope");
    std::string error;
    Require(runner->EndRequest(7, &error) && runner->EndRequest(8, &error),
            error);
  }

  // AR: greedy, sampled, stopped and cancelled requests on one pair.
  {
    Pair pair({}, {});
    const auto greedy = pair.Run(prompt, 6);
    Require(greedy.worker_error.empty(), "greedy: " + greedy.worker_error);
    Require(greedy.result.tokens.size() == 6 &&
                greedy.result.finish_reason == FinishReason::kLength,
            "greedy request produces its budget");
    pair.RequireSameCalls("greedy");
    const auto calls = pair.rank0().Log();
    Require(Count(calls, "prefill") == 3,
            "a 7-token prompt prefills in three chunks of three");
    // The last token is published without advancing it.
    Require(Count(calls, "advance") == 5, "greedy advances all but the last");
    Require(pair.rank0().cancellation_checks() == 0,
            "the model session never receives a cancellation check");

    const auto again = pair.Run(prompt, 6);
    Require(again.worker_error.empty() &&
                again.result.tokens == greedy.result.tokens,
            "a repeated request is reproducible: " + again.worker_error);
    pair.RequireSameCalls("repeated");

    const auto sampled = pair.Run(prompt, 8, Sampled());
    Require(sampled.worker_error.empty(), "sampled: " + sampled.worker_error);
    Require(sampled.result.tokens.size() == 8, "sampled request completes");
    pair.RequireSameCalls("sampled");

    // Stop at a generated piece that does not occur earlier in the output.
    std::optional<std::size_t> stop_at;
    std::string text;
    for (std::size_t i = 0; i < greedy.result.tokens.size(); ++i) {
      const auto piece = ToyRunner::Piece(greedy.result.tokens[i]);
      if (i >= 2 && text.find(piece) == std::string::npos) {
        stop_at = i;
        break;
      }
      text += piece;
    }
    Require(stop_at.has_value(), "the greedy output has a usable stop piece");
    const auto stop_piece = ToyRunner::Piece(greedy.result.tokens[*stop_at]);
    const auto stopped = pair.Run(prompt, 6, {}, {stop_piece});
    Require(stopped.worker_error.empty(), "stop: " + stopped.worker_error);
    Require(stopped.result.finish_reason == FinishReason::kStopSequence &&
                stopped.result.stop_sequence == stop_piece,
            "the request ends at its stop sequence");
    pair.RequireSameCalls("stop sequence");

    // Cancel after the second prefill chunk of a long prompt.
    const auto before = pair.rank0().prefill_calls();
    const std::vector<TextRunnerToken> long_prompt{1, 2,  3,  4,  5,  6,  7, 8,
                                                   9, 10, 11, 12, 13, 14, 15};
    const auto cancelled = pair.Run(long_prompt, 6, {}, {}, [&pair, before] {
      return pair.rank0().prefill_calls() >= before + 2;
    });
    Require(cancelled.worker_error.empty(),
            "cancel: " + cancelled.worker_error);
    Require(cancelled.result.cancelled ||
                cancelled.result.finish_reason == FinishReason::kCancelled,
            "the request is cancelled");
    Require(pair.rank0().prefill_calls() < before + 5,
            "cancellation stops prefill at a chunk boundary");
    pair.RequireSameCalls("cancelled during prefill");

    // A reset between requests reaches rank 1 too.
    pair.ResetBetweenRequests();
    const auto after_reset = pair.Run(prompt, 3);
    Require(after_reset.worker_error.empty(),
            "after an idle reset: " + after_reset.worker_error);
    Require(Count(pair.rank1().Log(), "reset") ==
                Count(pair.rank0().Log(), "reset") + 1,
            "rank 1 executed the reset sent between requests");
  }

  // Multi-token decoding is one instruction per cycle, greedy or sampled; a
  // sampled cycle carries rank 0's draw state.
  {
    Pair pair({.mtp = true}, {.mtp = true});
    const auto greedy = pair.Run(prompt, 7);
    Require(greedy.worker_error.empty(), "MTP greedy: " + greedy.worker_error);
    Require(greedy.result.tokens.size() == 7, "MTP greedy completes");
    Require(Count(pair.rank0().Log(), "decode") > 0,
            "greedy MTP decodes in multi-token cycles");
    pair.RequireSameCalls("MTP greedy");

    const auto decodes = Count(pair.rank0().Log(), "decode");
    const auto sampled = pair.Run(prompt, 9, Sampled());
    Require(sampled.worker_error.empty(),
            "MTP sampled: " + sampled.worker_error);
    Require(sampled.result.tokens.size() == 9, "MTP sampled completes");
    Require(Count(pair.rank0().Log(), "decode") > decodes,
            "sampled MTP decodes in multi-token cycles too");
    pair.RequireSameCalls("MTP sampled");
    const auto again = pair.Run(prompt, 9, Sampled());
    Require(
        again.worker_error.empty() &&
            again.result.tokens == sampled.result.tokens,
        "a seeded sampled MTP request is reproducible: " + again.worker_error);
    pair.RequireSameCalls("MTP sampled again");

    // Without a seed each rank's sampler starts from its own random state, so
    // only rank 0's draw state, carried by each decode, keeps them together.
    auto unseeded = Sampled();
    unseeded.seed = -1;
    const auto random_draws = pair.Run(prompt, 9, unseeded);
    Require(random_draws.worker_error.empty(),
            "unseeded sampled MTP: " + random_draws.worker_error);
    pair.RequireSameCalls("MTP sampled without a seed");
  }

  // Rank 1's own greedy choice catches a numerical divergence.
  {
    Pair pair({}, {.diverge_at = prompt.size() + 2});
    const auto diverged = pair.Run(prompt, 6);
    Require(diverged.worker_error.find("selected token") != std::string::npos,
            "a rank-1 divergence fails the request: " + diverged.worker_error);
    // Rank 1 still advanced rank 0's tokens, so the pair stays usable.
    pair.RequireSameCalls("after divergence");
    const auto next = pair.Run(prompt, 6);
    Require(next.worker_error.empty(),
            "the next request succeeds: " + next.worker_error);
  }

  // What a rejected request computed is never reused: the next request
  // recomputes it, and that agreed result is reusable again.
  {
    Pair pair({.cache = true},
              {.cache = true, .diverge_at = prompt.size() + 2});
    const auto diverged = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(diverged.worker_error.find("selected token") != std::string::npos,
            "a rank-1 divergence fails the request: " + diverged.worker_error);
    const auto recomputed = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(recomputed.worker_error.empty() &&
                recomputed.result.cached_prompt_tokens == 0,
            "a rejected request's checkpoint is not reused: " +
                recomputed.worker_error);
    const auto reused = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(reused.worker_error.empty() &&
                reused.result.cached_prompt_tokens == prompt.size() &&
                reused.result.tokens == recomputed.result.tokens,
            "an agreed recomputation is reused: " + reused.worker_error);
    pair.RequireSameCalls("rejected continuation");
  }

  // A request reuses what the previous one left before rank 1's verdict on
  // it arrives, as on one host, and stands or falls with that verdict.
  for (const bool reject : {false, true}) {
    const std::string what =
        reject ? "reuse before a rejection" : "reuse before a verdict";
    ToyOptions rank1{.cache = true};
    if (reject) {
      rank1.diverge_at = prompt.size() + 2;
    }
    Pair pair({.cache = true}, rank1);
    auto first = pair.RunOpen(prompt, 6, {}, {}, {}, true);
    auto second = pair.RunOpen(prompt, 6, {}, {}, {}, true);
    Require(second.outcome.result.cached_prompt_tokens == prompt.size(),
            what + ": the second request reuses the first one's prompt");
    const auto first_end = pair.Close(std::move(first));
    const auto second_end = pair.Close(std::move(second));
    Require(first_end.worker_error.empty() != reject,
            what + ": the first request's verdict: " + first_end.worker_error);
    Require(
        second_end.worker_error.empty() != reject,
        what + ": the second request follows it: " + second_end.worker_error);
  }

  // A producer still open may outlast any grace: only one that ended and has
  // no verdict is overdue (a long-running request must not fail its reuser).
  {
    Pair pair({.cache = true}, {.cache = true});
    pair.SetDependencyGrace(std::chrono::milliseconds(20));
    auto first = pair.RunOpen(prompt, 6, {}, {}, {}, true);
    auto second = pair.RunOpen(prompt, 6, {}, {}, {}, true);
    Require(second.outcome.result.cached_prompt_tokens == prompt.size(),
            "the reuse the wait is about");
    auto second_end = std::async(std::launch::async,
                                 [&] { return pair.Close(std::move(second)); });
    Require(second_end.wait_for(std::chrono::milliseconds(200)) ==
                std::future_status::timeout,
            "a reuser outlasts the grace while its producer stays open");
    const auto first_end = pair.Close(std::move(first));
    const auto closed = second_end.get();
    Require(first_end.worker_error.empty() && closed.worker_error.empty(),
            "the reuser stands with its producer: " + first_end.worker_error +
                closed.worker_error);
  }
  // A producer that ended and never gets a verdict is overdue.
  {
    Pair pair({.cache = true}, {.cache = true});
    pair.SetDependencyGrace(std::chrono::milliseconds(20));
    auto first = pair.RunOpen(prompt, 6, {}, {}, {}, true);
    auto second = pair.RunOpen(prompt, 6, {}, {}, {}, true);
    pair.EndWithoutVerdict(first);
    const auto closed = pair.Close(std::move(second));
    Require(closed.worker_error == "reused a rejected state",
            "an overdue verdict fails the reuser: " + closed.worker_error);
  }

  // Concurrent requests: each call names its request, concurrent decoders run
  // in one batched instruction, greedy MTP ones with rank 0's draft plan
  // (rank 1 would choose another), and every request produces what it
  // produces alone.
  const std::vector<TextRunnerToken> other{2, 4, 6, 8, 10, 12, 14};
  for (const auto& [what, mtp, sampled] :
       {std::tuple{"concurrent AR", false, false},
        std::tuple{"concurrent sampled MTP", true, true},
        std::tuple{"concurrent greedy MTP", true, false}}) {
    const auto sampling =
        sampled ? Sampled() : gufo::sampling::SamplingConfig{};
    std::vector<TextRunnerToken> alone[2];
    {
      Pair solo({.mtp = mtp}, {.mtp = mtp});
      alone[0] = solo.Run(prompt, 12, sampling).result.tokens;
      alone[1] = solo.Run(other, 12, sampling).result.tokens;
    }
    SubmissionGate gate(2);
    Pair pair(
        {.mtp = mtp, .prefill_hook = gate.Hook(), .width = 4, .batch_plan = 1},
        {.mtp = mtp, .width = 4, .batch_plan = 2}, 2);
    const auto submitted = [&gate] { gate.Submitted(); };
    auto first = std::async(std::launch::async, [&] {
      return pair.Run(prompt, 12, sampling, {}, {}, false, 0, submitted);
    });
    auto second = std::async(std::launch::async, [&] {
      return pair.Run(other, 12, sampling, {}, {}, false, 0, submitted);
    });
    const auto a = first.get();
    const auto b = second.get();
    Require(a.worker_error.empty() && b.worker_error.empty(),
            std::string(what) + ": " + a.worker_error + b.worker_error);
    Require(a.result.tokens == alone[0] && b.result.tokens == alone[1],
            std::string(what) + ": requests produce what they produce alone");
    const auto log = pair.rank0().Log();
    Require(Count(log, mtp ? "decode batch 2" : "batch 2") > 0,
            std::string(what) + ": decoders run in batches");
    Require(!mtp || Count(log, std::string("decode batch 2 plan ") +
                                   (sampled ? "own" : "1")) ==
                        Count(log, "decode batch"),
            std::string(what) +
                ": a greedy batch drafts by rank 0's plan, a sampled one by "
                "each member's history");
    pair.RequireSameCalls(what);
    Require(pair.open_requests() == 0,
            std::string(what) + ": rank 1 closed both requests");
  }

  // The execution digest catches a result the ranks disagree on.
  {
    Pair pair({.mtp = true}, {.mtp = true, .extra_draft_once = true});
    const auto mismatch = pair.Run(prompt, 7);
    Require(mismatch.worker_error.find("different model calls") !=
                std::string::npos,
            "a result difference fails the request: " + mismatch.worker_error);
    const auto next = pair.Run(prompt, 7);
    Require(next.worker_error.empty(),
            "the next request succeeds: " + next.worker_error);
  }

  // A request with images: rank 1 rebuilds its prompt context and binds it
  // wherever rank 0 does, and the cache keeps image and text states apart.
  for (const bool mtp : {false, true}) {
    const std::string what = mtp ? "MTP images" : "AR images";
    Pair pair({.mtp = mtp, .cache = true}, {.mtp = mtp, .cache = true});
    const auto image = std::make_shared<ToyContext>(7);
    const auto text = pair.Run(prompt, 6, {}, {}, {}, true);
    const auto seen = pair.Run(prompt, 6, {}, {}, {}, true, 0, {}, image);
    const auto again = pair.Run(prompt, 6, {}, {}, {}, true, 0, {}, image);
    const auto text_again = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(text.worker_error.empty() && seen.worker_error.empty() &&
                again.worker_error.empty() && text_again.worker_error.empty(),
            what + ": " + text.worker_error + seen.worker_error +
                again.worker_error + text_again.worker_error);
    Require(seen.result.tokens != text.result.tokens,
            what + ": the image changes the output");
    Require(seen.result.cached_prompt_tokens == 0,
            what + ": an image request does not reuse a text state");
    Require(again.result.tokens == seen.result.tokens &&
                again.result.cached_prompt_tokens == prompt.size(),
            what + ": the same image reuses its prompt");
    Require(text_again.result.tokens == text.result.tokens,
            what + ": text after an image matches text alone");
    Require(Count(pair.rank0().Log(), "context") > 0,
            what + ": the image context reaches the model");
    pair.RequireSameCalls(what);
  }

  // The disk cache: rank 0 persists both ranks' state in one entry, and a
  // restart of the pair restores both from it.
  for (const bool mtp : {false, true}) {
    const std::string what = mtp ? "MTP disk cache" : "AR disk cache";
    const auto root = std::filesystem::temp_directory_path() /
                      ("tp2-disk-test-" + std::to_string(::getpid()) +
                       (mtp ? "-mtp" : "-ar"));
    std::filesystem::remove_all(root);
    const ToyOptions options{.mtp = mtp, .cache = true};
    std::vector<TextRunnerToken> first;
    {
      Pair pair(options, options, 1, root);
      const auto run = pair.Run(prompt, 6, {}, {}, {}, true);
      Require(run.worker_error.empty(), what + ": " + run.worker_error);
      first = run.result.tokens;
      WaitForDisk(root);
      pair.RequireSameCalls(what + " save");
    }
    {
      Pair pair(options, options, 1, root);
      const auto run = pair.Run(prompt, 6, {}, {}, {}, true);
      Require(run.worker_error.empty(), what + ": " + run.worker_error);
      Require(run.result.tokens == first && run.result.cache_disk_hit &&
                  run.result.cached_prompt_tokens > 0,
              what + ": a restart restores both ranks from disk");
      pair.RequireSameCalls(what + " restore");
    }
    std::filesystem::remove_all(root);
  }

  // A prompt context rank 1 cannot rebuild fails that request only.
  {
    Pair pair({}, {.fail_context_decode = true});
    const auto image = std::make_shared<ToyContext>(3);
    const auto refused = pair.Run(prompt, 4, {}, {}, {}, false, 0, {}, image);
    Require(refused.worker_error.find("prompt context") != std::string::npos,
            "an unreadable prompt context fails the request: " +
                refused.worker_error);
    const auto next = pair.Run(prompt, 4);
    Require(next.worker_error.empty(),
            "the next request succeeds: " + next.worker_error);
  }

  std::puts(
      "PASS: TP2 executor mirrors greedy, sampled, stopped, cancelled, "
      "multi-token, concurrent, batched and image requests, reports divergence "
      "and fails requests that reused a rejected continuation");
  return 0;
}
