#include "src/models/qwen38_flash_next/distributed/tp_fault.hpp"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

namespace gufo::models::qwen38_flash_next::distributed {
namespace {

constexpr std::string_view kPeerLostPrefix = "peer lost: ";
constexpr std::string_view kPeerPoisonedPrefix = "peer poisoned: ";

void PutWord(PoisonFrame* frame, std::size_t offset, std::uint32_t value) {
  const std::uint32_t wire = htonl(value);
  std::memcpy(frame->data() + offset, &wire, sizeof(wire));
}

std::uint32_t GetWord(std::span<const std::uint8_t> frame, std::size_t offset) {
  std::uint32_t wire = 0;
  std::memcpy(&wire, frame.data() + offset, sizeof(wire));
  return ntohl(wire);
}

TpFault Lost(std::string reason) {
  return {.kind = TpFaultKind::kPeerLost, .reason = std::move(reason)};
}

}  // namespace

std::string DescribeTpFault(const TpFault& fault) {
  switch (fault.kind) {
    case TpFaultKind::kPeerLost:
      return std::string(kPeerLostPrefix) + fault.reason;
    case TpFaultKind::kPeerPoisoned:
      return std::string(kPeerPoisonedPrefix) + fault.reason;
    case TpFaultKind::kNone:
    case TpFaultKind::kLocal:
      break;
  }
  return fault.reason;
}

bool IsPeerFaultMessage(std::string_view message) {
  return message.find(kPeerLostPrefix) != std::string_view::npos ||
         message.find(kPeerPoisonedPrefix) != std::string_view::npos;
}

PoisonFrame EncodePoisonFrame(std::uint32_t rank, std::string_view reason) {
  PoisonFrame frame{};
  const std::size_t length = std::min(reason.size(), kPoisonReasonBytes);
  PutWord(&frame, 0, kPoisonMagic);
  PutWord(&frame, 4, kPoisonVersion);
  PutWord(&frame, 8, rank);
  PutWord(&frame, 12, static_cast<std::uint32_t>(length));
  std::memcpy(frame.data() + 16, reason.data(), length);
  return frame;
}

std::optional<std::string> DecodePoisonFrame(
    std::span<const std::uint8_t> frame) {
  if (frame.size() != kPoisonFrameBytes ||
      GetWord(frame, 0) != kPoisonMagic ||
      GetWord(frame, 4) != kPoisonVersion) {
    return std::nullopt;
  }
  const std::uint32_t length = GetWord(frame, 12);
  if (length > kPoisonReasonBytes) {
    return std::nullopt;
  }
  return "rank " + std::to_string(GetWord(frame, 8)) + ": " +
         std::string(reinterpret_cast<const char*>(frame.data() + 16), length);
}

bool SendPoisonFrame(int fd, std::uint32_t rank, std::string_view reason) {
  if (fd < 0) {
    return false;
  }
  const PoisonFrame frame = EncodePoisonFrame(rank, reason);
  ssize_t sent = -1;
  do {
    sent = ::send(fd, frame.data(), frame.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
  } while (sent < 0 && errno == EINTR);
  return sent == static_cast<ssize_t>(frame.size());
}

TpFault ReadPeerFault(int fd, std::chrono::milliseconds wait) {
  PoisonFrame frame{};
  std::size_t got = 0;
  const auto deadline = std::chrono::steady_clock::now() + wait;
  for (;;) {
    const ssize_t n =
        ::recv(fd, frame.data() + got, frame.size() - got, MSG_DONTWAIT);
    if (n > 0) {
      got += static_cast<std::size_t>(n);
      if (got == frame.size()) {
        break;
      }
      continue;
    }
    if (n == 0) {
      return Lost(got == 0
                      ? "the peer closed the bootstrap connection (its "
                        "process exited)"
                      : "the peer closed the bootstrap connection inside a "
                        "frame");
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      if (got == 0) {
        return {};
      }
      const auto remaining = std::chrono::duration_cast<
          std::chrono::milliseconds>(deadline -
                                     std::chrono::steady_clock::now());
      if (remaining.count() <= 0) {
        return Lost("the peer sent a partial frame on the bootstrap "
                    "connection");
      }
      pollfd poll_fd{.fd = fd, .events = POLLIN, .revents = 0};
      (void)::poll(&poll_fd, 1, static_cast<int>(remaining.count()));
      continue;
    }
    if (errno == ETIMEDOUT) {
      return Lost("the bootstrap connection's keepalive timed out (the peer "
                  "host is gone or unreachable)");
    }
    if (errno == ECONNRESET || errno == EPIPE) {
      return Lost("the peer reset the bootstrap connection");
    }
    return Lost(std::string("bootstrap connection error: ") +
                std::strerror(errno));
  }
  if (const auto reason = DecodePoisonFrame(frame)) {
    return {.kind = TpFaultKind::kPeerPoisoned, .reason = *reason};
  }
  return Lost("unexpected bytes on the bootstrap connection");
}

TpFault ClassifyCompletionStatus(int status, std::string_view status_text) {
  const std::string head = "completion status " + std::to_string(status) +
                           " (" + std::string(status_text) + ")";
  switch (status) {
    case kWcRetryExcErr:
      return Lost(head +
                  ": the peer's NIC acknowledged nothing within the QP's "
                  "retries (the peer rank, its host or the link is gone)");
    case kWcRnrRetryExcErr:
      return Lost(head + ": the peer posted no receive within the RNR "
                         "retries (the peer rank stopped)");
    case kWcWrFlushErr:
      return Lost(head + ": the queue pair had already failed");
    case kWcRemInvReqErr:
    case kWcRemAccessErr:
    case kWcRemOpErr:
    case kWcRemInvRdReqErr:
    case kWcRemAbortErr:
      return Lost(head + ": the peer's NIC refused the request (its queue "
                         "pair or memory region is gone)");
    default:
      break;
  }
  return {.kind = TpFaultKind::kLocal, .reason = head};
}

}  // namespace gufo::models::qwen38_flash_next::distributed
