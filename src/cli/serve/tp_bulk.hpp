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
  /// Rank 1's cache rows or private state, answering a control instruction.
  kOut = 2,
  /// Rows or private state for rank 1's slot. Rank 0 sends these only between
  /// an acknowledged BeginRestore and its Validate.
  kIn = 3,
  /// Rank 1 settled a `kIn`.
  kInDone = 4,
  /// Rank 1 could not produce a `kOut` or settle a `kIn`. `bytes` is a message.
  kFailed = 5,
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
/// Rank 0's single cache owns both ranks' retained state, so rank 1's half of
/// every checkpoint crosses here. It is apart from the control channel so that
/// a restore's gigabytes never delay the instructions of other requests.
/// Connections prove the control token as the control channel does. One thread
/// may send and another receive at once; sends are serialized.
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
