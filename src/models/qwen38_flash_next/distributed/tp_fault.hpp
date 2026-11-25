#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_FAULT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_FAULT_HPP_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

// The TP2 fault contract's host-side pieces (TP2.md, "Fault contract"): the
// poison frame a rank writes to its peer at its first failure, how the peer
// reads what woke its bootstrap socket, and how a failed work completion is
// named. No verbs or HIP header, so the CPU suite tests them.
namespace gufo::models::qwen38_flash_next::distributed {

enum class TpFaultKind {
  /// Nothing to report (a wake-up with no data).
  kNone,
  /// The peer rank or its host is gone: its socket closed or reset, its
  /// keepalive failed, or its NIC stopped acknowledging.
  kPeerLost,
  /// The peer failed first and said why in a poison frame.
  kPeerPoisoned,
  /// This rank's own failure.
  kLocal,
};

struct TpFault {
  TpFaultKind kind{TpFaultKind::kNone};
  std::string reason;
};

/// "peer lost: <reason>", "peer poisoned: <reason>" or the reason alone. The
/// prefixes are stable: logs and the kill-a-rank check match on them.
[[nodiscard]] std::string DescribeTpFault(const TpFault& fault);

/// True for a message DescribeTpFault made from a peer fault: a rank does not
/// poison its peer back with the peer's own failure.
[[nodiscard]] bool IsPeerFaultMessage(std::string_view message);

inline constexpr std::size_t kPoisonFrameBytes = 64;
inline constexpr std::uint32_t kPoisonMagic = 0x47555058U;  // "GUPX"
inline constexpr std::uint32_t kPoisonVersion = 1;
/// The reason's bytes in a frame: the rest of 64 after four words.
inline constexpr std::size_t kPoisonReasonBytes = kPoisonFrameBytes - 16;

using PoisonFrame = std::array<std::uint8_t, kPoisonFrameBytes>;

/// Magic, version, rank and reason length as big-endian words, then the
/// reason, cut to kPoisonReasonBytes.
[[nodiscard]] PoisonFrame EncodePoisonFrame(std::uint32_t rank,
                                            std::string_view reason);

/// "rank <n>: <reason>" for a valid frame, nullopt otherwise.
[[nodiscard]] std::optional<std::string> DecodePoisonFrame(
    std::span<const std::uint8_t> frame);

/// Writes the poison frame without blocking and without SIGPIPE. False when
/// the socket refused it, e.g. because the peer is gone already; the caller
/// has failed either way and does not retry.
bool SendPoisonFrame(int fd, std::uint32_t rank, std::string_view reason);

/// Reads what made the bootstrap socket readable (it carries nothing else
/// after setup), waiting at most `wait` for the rest of a frame: a poison
/// frame is kPeerPoisoned; end of file, a reset, a keepalive timeout, any
/// other error or unexpected bytes are kPeerLost; no data is kNone.
[[nodiscard]] TpFault ReadPeerFault(int fd, std::chrono::milliseconds wait);

// ibv_wc_status values (infiniband/verbs.h; verbs.cpp checks them).
inline constexpr int kWcWrFlushErr = 5;
inline constexpr int kWcRemInvReqErr = 9;
inline constexpr int kWcRemAccessErr = 10;
inline constexpr int kWcRemOpErr = 11;
inline constexpr int kWcRetryExcErr = 12;
inline constexpr int kWcRnrRetryExcErr = 13;
inline constexpr int kWcRemInvRdReqErr = 15;
inline constexpr int kWcRemAbortErr = 16;

/// Names a failed work completion. RETRY_EXC (no acknowledgement within the
/// QP's retries), RNR_RETRY_EXC (no receive posted), a flush (the QP failed
/// already) and the remote errors are kPeerLost; anything else is kLocal.
/// `status_text` is ibv_wc_status_str's.
[[nodiscard]] TpFault ClassifyCompletionStatus(int status,
                                               std::string_view status_text);

}  // namespace gufo::models::qwen38_flash_next::distributed

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_FAULT_HPP_
