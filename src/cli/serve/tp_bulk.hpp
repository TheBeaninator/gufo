#ifndef GUFO_SERVER_TP_BULK_HPP_
#define GUFO_SERVER_TP_BULK_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>

namespace gufo::server {

/// What one frame of the bulk connection carries.
enum class TpBulkKind : std::uint32_t {
  /// Rank 1's component descriptors, once, when the connection opens.
  kHello = 1,
  /// Rank 0's cache dropped its RAM copy of stored entry `transfer` (the key);
  /// rank 0 sends these, without a payload. (2 and 3 answered transfers before
  /// protocol 24.)
  kRelease = 4,
  /// Rank 0 published one of its disk files: `is_private` and the payload's
  /// 16-byte file id, then the 8-byte keys its references name.
  kPublished = 5,
  /// Rank 0 unlinked a disk file: `is_private` and the 16-byte file id.
  kRemoved = 6,
  /// Rank 0's disk files at its startup, 17 bytes each: id, then 1 if private.
  kLive = 7,
};

struct TpBulkHeader {
  TpBulkKind kind{TpBulkKind::kHello};
  std::uint32_t state{0};
  std::uint32_t component{0};
  /// 1 for private state, whose `first` is a byte offset and `end` its end.
  std::uint32_t is_private{0};
  std::uint64_t transfer{0};
  std::uint64_t first{0};
  std::uint64_t end{0};
  std::uint64_t bytes{0};
};

/// The second TCP connection of a TP2 pair, next to the control channel.
///
/// Rank 0's cache retains rank 1's half of each checkpoint only as keys into
/// rank 1's own store; this connection carries the completions of stashes and
/// loads, and rank 0's releases, which its cache issues from destructors and
/// so cannot wait behind the control channel's instructions. Connections prove
/// the control token as the control channel does. One thread may send and
/// another receive at once; sends are serialized.
class TpBulkSocket final {
public:
  [[nodiscard]] static std::shared_ptr<TpBulkSocket> Listen(
      std::uint16_t port, const std::string& token, std::string* error);
  [[nodiscard]] static std::shared_ptr<TpBulkSocket> Connect(
      const std::string& host, std::uint16_t port, const std::string& token,
      std::string* error);
  /// A connected pair within one process, for tests.
  [[nodiscard]] static bool Pair(std::shared_ptr<TpBulkSocket>* rank0,
                                 std::shared_ptr<TpBulkSocket>* rank1,
                                 std::string* error);
  ~TpBulkSocket();
  TpBulkSocket(const TpBulkSocket&) = delete;
  TpBulkSocket& operator=(const TpBulkSocket&) = delete;

  /// Sends a header and its `bytes`-sized payload as one frame.
  [[nodiscard]] bool Send(const TpBulkHeader& header,
                          std::span<const std::byte> payload,
                          std::string* error);
  /// Blocks for the next frame. An idle connection is normal, so only a lost
  /// peer or `Shutdown` ends the wait.
  [[nodiscard]] bool ReceiveHeader(TpBulkHeader* header, std::string* error);
  /// Receives the frame's payload, all of it in one or several calls.
  [[nodiscard]] bool ReceivePayload(std::span<std::byte> payload,
                                    std::string* error);
  /// Wakes both directions; every later call fails.
  void Shutdown() noexcept;

private:
  explicit TpBulkSocket(int fd);

  const int fd_;
  std::mutex send_mutex_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TP_BULK_HPP_
