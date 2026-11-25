#include "src/models/qwen38_flash_next/distributed/tp_fault.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {

namespace d = gufo::models::qwen38_flash_next::distributed;
using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

int failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

struct Pair {
  int a{-1};
  int b{-1};
  Pair() {
    int fds[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
      std::perror("socketpair");
      std::exit(2);
    }
    a = fds[0];
    b = fds[1];
  }
  ~Pair() {
    if (a >= 0) ::close(a);
    if (b >= 0) ::close(b);
  }
  void CloseA() {
    ::close(a);
    a = -1;
  }
};

long ElapsedMs(Clock::time_point start) {
  return static_cast<long>(
      std::chrono::duration_cast<milliseconds>(Clock::now() - start).count());
}

void FrameRoundTrip() {
  const auto frame = d::EncodePoisonFrame(1, "verbs collective identity mismatch");
  const auto decoded = d::DecodePoisonFrame(frame);
  Check(decoded == std::string("rank 1: verbs collective identity mismatch"),
        "frame round trip: " + decoded.value_or("<none>"));
  const std::string long_reason(200, 'x');
  const auto cut = d::DecodePoisonFrame(d::EncodePoisonFrame(0, long_reason));
  Check(cut.has_value() && cut->size() == std::string("rank 0: ").size() +
                                              d::kPoisonReasonBytes,
        "a long reason is cut to the frame");
  auto bad = frame;
  bad[0] ^= 0xff;
  Check(!d::DecodePoisonFrame(bad).has_value(), "bad magic refused");
  bad = frame;
  bad[15] = 200;  // reason length past the frame
  Check(!d::DecodePoisonFrame(bad).has_value(), "bad length refused");
  Check(!d::DecodePoisonFrame(std::span(frame.data(), 63)).has_value(),
        "short frame refused");
}

void PoisonReachesThePeer() {
  Pair pair;
  Check(d::SendPoisonFrame(pair.a, 1, "a GPU wait timed out"), "poison sent");
  const auto fault = d::ReadPeerFault(pair.b, milliseconds(100));
  Check(fault.kind == d::TpFaultKind::kPeerPoisoned,
        "poison frame read as kPeerPoisoned");
  const auto text = d::DescribeTpFault(fault);
  Check(text == "peer poisoned: rank 1: a GPU wait timed out", "described: " + text);
  Check(d::IsPeerFaultMessage("verbs " + text), "a peer fault is recognised");
}

void QuietSocketIsNone() {
  Pair pair;
  const auto start = Clock::now();
  const auto fault = d::ReadPeerFault(pair.b, milliseconds(500));
  Check(fault.kind == d::TpFaultKind::kNone, "no data is kNone");
  Check(ElapsedMs(start) < 100, "kNone returns without waiting");
}

void ClosedPeerIsLost() {
  Pair pair;
  pair.CloseA();
  const auto fault = d::ReadPeerFault(pair.b, milliseconds(100));
  Check(fault.kind == d::TpFaultKind::kPeerLost &&
            fault.reason.find("closed") != std::string::npos,
        "closed peer is kPeerLost: " + fault.reason);
  Check(d::DescribeTpFault(fault).rfind("peer lost: ", 0) == 0, "lost prefix");
  // Sending poison to a gone peer fails quietly: no SIGPIPE ends this test.
  Check(!d::SendPoisonFrame(pair.b, 0, "late"), "poison to a gone peer is refused");
}

void PartialFrameBounded() {
  {
    Pair pair;
    const auto frame = d::EncodePoisonFrame(1, "split");
    Check(::send(pair.a, frame.data(), 10, 0) == 10, "first part sent");
    const auto start = Clock::now();
    const auto fault = d::ReadPeerFault(pair.b, milliseconds(50));
    Check(fault.kind == d::TpFaultKind::kPeerLost &&
              fault.reason.find("partial") != std::string::npos,
          "a partial frame that never completes is kPeerLost: " + fault.reason);
    Check(ElapsedMs(start) < 1000, "the partial-frame wait is bounded");
  }
  {
    Pair pair;
    const auto frame = d::EncodePoisonFrame(1, "split");
    Check(::send(pair.a, frame.data(), 10, 0) == 10, "first part sent");
    std::thread rest([&] {
      std::this_thread::sleep_for(milliseconds(20));
      (void)::send(pair.a, frame.data() + 10, frame.size() - 10, 0);
    });
    const auto fault = d::ReadPeerFault(pair.b, milliseconds(1000));
    rest.join();
    Check(fault.kind == d::TpFaultKind::kPeerPoisoned,
          "a frame completed within the wait is read whole: " + fault.reason);
  }
  {
    Pair pair;
    char junk[64];
    std::memset(junk, 'j', sizeof(junk));
    Check(::send(pair.a, junk, sizeof(junk), 0) == 64, "junk sent");
    const auto fault = d::ReadPeerFault(pair.b, milliseconds(50));
    Check(fault.kind == d::TpFaultKind::kPeerLost &&
              fault.reason.find("unexpected") != std::string::npos,
          "junk is kPeerLost: " + fault.reason);
  }
}

void CompletionStatuses() {
  const auto retry = d::ClassifyCompletionStatus(d::kWcRetryExcErr,
                                                 "transport retry counter exceeded");
  Check(retry.kind == d::TpFaultKind::kPeerLost, "RETRY_EXC is peer lost");
  Check(d::DescribeTpFault(retry).rfind(
            "peer lost: completion status 12 (transport retry counter exceeded)",
            0) == 0,
        "RETRY_EXC described: " + d::DescribeTpFault(retry));
  for (const int status : {d::kWcRnrRetryExcErr, d::kWcWrFlushErr,
                           d::kWcRemAccessErr, d::kWcRemOpErr,
                           d::kWcRemInvReqErr, d::kWcRemInvRdReqErr,
                           d::kWcRemAbortErr}) {
    Check(d::ClassifyCompletionStatus(status, "x").kind ==
              d::TpFaultKind::kPeerLost,
          "status " + std::to_string(status) + " is peer lost");
  }
  const auto local = d::ClassifyCompletionStatus(4, "local protection error");
  Check(local.kind == d::TpFaultKind::kLocal, "LOC_PROT is local");
  Check(!d::IsPeerFaultMessage(d::DescribeTpFault(local)),
        "a local fault is not a peer fault");
}

/// The kill-a-rank contract on the host: a child process holds the peer end
/// of a loopback TCP connection set up like the bootstrap socket; SIGKILL it
/// while this "rank" sleeps in poll with a 30 s exchange deadline. The wait
/// must end with kPeerLost within milliseconds, not at the deadline.
void KilledRankIsLostWithinMilliseconds() {
  const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t size = sizeof(address);
  if (listener < 0 ||
      ::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
      ::listen(listener, 1) != 0 ||
      ::getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size) != 0) {
    std::perror("listen");
    std::exit(2);
  }
  const pid_t child = ::fork();
  if (child == 0) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
      ::_exit(3);
    }
    for (;;) {
      ::pause();
    }
  }
  const int peer = ::accept(listener, nullptr, nullptr);
  ::close(listener);
  const int on = 1;
  (void)::setsockopt(peer, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on));
  std::this_thread::sleep_for(milliseconds(50));
  const auto killed = Clock::now();
  ::kill(child, SIGKILL);
  pollfd fd{.fd = peer, .events = POLLIN, .revents = 0};
  const int ready = ::poll(&fd, 1, 30000);
  const auto fault = d::ReadPeerFault(peer, milliseconds(100));
  const long ms = ElapsedMs(killed);
  int status = 0;
  ::waitpid(child, &status, 0);
  ::close(peer);
  Check(ready == 1 && fault.kind == d::TpFaultKind::kPeerLost,
        "killed rank is kPeerLost: " + d::DescribeTpFault(fault));
  Check(ms < 1000, "killed rank detected in " + std::to_string(ms) + " ms (< 1000)");
  std::printf("killed rank detected in %ld ms: %s\n", ms,
              d::DescribeTpFault(fault).c_str());
}

}  // namespace

int main() {
  FrameRoundTrip();
  PoisonReachesThePeer();
  QuietSocketIsNone();
  ClosedPeerIsLost();
  PartialFrameBounded();
  CompletionStatuses();
  KilledRankIsLostWithinMilliseconds();
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("tp_fault_test: all checks passed\n");
  return 0;
}
