#include "src/cli/serve/tp_bulk.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace gufo::server {
namespace {

constexpr std::size_t kNonceBytes = 32;
constexpr std::size_t kMacBytes = 32;
constexpr auto kPairTimeout = std::chrono::seconds(60);
/// Large windows carry a checkpoint at the link's rate over its latency.
constexpr int kSocketBufferBytes = 16 << 20;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string SystemError(const char* operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

void Tune(int fd) {
  const int on = 1;
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
  (void)::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on));
  (void)::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &kSocketBufferBytes,
                     sizeof(kSocketBufferBytes));
  (void)::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &kSocketBufferBytes,
                     sizeof(kSocketBufferBytes));
}

bool SendAll(int fd, const void* data, std::size_t bytes, std::string* error) {
  const auto* cursor = static_cast<const std::uint8_t*>(data);
  while (bytes != 0) {
    const auto sent = ::send(fd, cursor, bytes, MSG_NOSIGNAL);
    if (sent < 0 && errno == EINTR) {
      continue;
    }
    if (sent <= 0) {
      SetError(error, SystemError("TP bulk send"));
      return false;
    }
    cursor += sent;
    bytes -= static_cast<std::size_t>(sent);
  }
  return true;
}

bool RecvAll(int fd, void* data, std::size_t bytes, std::string* error) {
  auto* cursor = static_cast<std::uint8_t*>(data);
  while (bytes != 0) {
    const auto received = ::recv(fd, cursor, bytes, 0);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received <= 0) {
      SetError(error, received == 0 ? "TP bulk peer closed the connection"
                                    : SystemError("TP bulk receive"));
      return false;
    }
    cursor += received;
    bytes -= static_cast<std::size_t>(received);
  }
  return true;
}

/// HMAC-SHA256 keyed by the control token over the sender's rank and both
/// nonces, so a MAC answers only this connection and cannot be reflected.
bool BulkMac(const std::string& token, std::uint8_t sender,
             const std::array<std::uint8_t, kNonceBytes>& rank0,
             const std::array<std::uint8_t, kNonceBytes>& rank1,
             std::array<std::uint8_t, kMacBytes>* mac) {
  constexpr std::string_view kLabel = "gufo tp2 bulk v1";
  std::vector<std::uint8_t> message(kLabel.begin(), kLabel.end());
  message.push_back(sender);
  message.insert(message.end(), rank0.begin(), rank0.end());
  message.insert(message.end(), rank1.begin(), rank1.end());
  unsigned int size = 0;
  return HMAC(EVP_sha256(), token.data(), static_cast<int>(token.size()),
              message.data(), message.size(), mac->data(), &size) != nullptr &&
         size == kMacBytes;
}

bool Authenticate(int fd, const std::string& token, std::uint8_t rank,
                  std::string* error) {
  std::array<std::uint8_t, kNonceBytes> own{}, peer{};
  if (RAND_bytes(own.data(), static_cast<int>(own.size())) != 1) {
    SetError(error, "TP bulk nonce generation failed");
    return false;
  }
  if (!SendAll(fd, own.data(), own.size(), error) ||
      !RecvAll(fd, peer.data(), peer.size(), error)) {
    return false;
  }
  const auto& rank0 = rank == 0 ? own : peer;
  const auto& rank1 = rank == 0 ? peer : own;
  std::array<std::uint8_t, kMacBytes> mine{}, expected{}, theirs{};
  if (!BulkMac(token, rank, rank0, rank1, &mine) ||
      !BulkMac(token, rank == 0 ? 1 : 0, rank0, rank1, &expected)) {
    SetError(error, "TP bulk authentication failed");
    return false;
  }
  if (!SendAll(fd, mine.data(), mine.size(), error) ||
      !RecvAll(fd, theirs.data(), theirs.size(), error)) {
    return false;
  }
  if (CRYPTO_memcmp(theirs.data(), expected.data(), expected.size()) != 0) {
    SetError(error, "TP bulk peer does not hold the control token");
    return false;
  }
  return true;
}

}  // namespace

TpBulkSocket::TpBulkSocket(int fd) : fd_(fd) {
  Tune(fd_);
}

TpBulkSocket::~TpBulkSocket() {
  ::close(fd_);
}

std::shared_ptr<TpBulkSocket> TpBulkSocket::Listen(std::uint16_t port,
                                                   const std::string& token,
                                                   std::string* error) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    SetError(error, SystemError("TP bulk socket"));
    return nullptr;
  }
  const int on = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
  // Set before accept: the window scale is agreed during the handshake.
  (void)::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &kSocketBufferBytes,
                     sizeof(kSocketBufferBytes));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);
  if (::bind(fd, reinterpret_cast<const sockaddr*>(&address),
             sizeof(address)) != 0 ||
      ::listen(fd, 1) != 0) {
    SetError(error, SystemError("TP bulk listen"));
    ::close(fd);
    return nullptr;
  }
  pollfd descriptor{.fd = fd, .events = POLLIN, .revents = 0};
  const auto waited =
      ::poll(&descriptor, 1,
             std::chrono::duration_cast<std::chrono::milliseconds>(kPairTimeout)
                 .count());
  if (waited <= 0) {
    SetError(error, waited == 0 ? "TP bulk accept timed out"
                                : SystemError("TP bulk poll"));
    ::close(fd);
    return nullptr;
  }
  const int peer = ::accept(fd, nullptr, nullptr);
  ::close(fd);
  if (peer < 0) {
    SetError(error, SystemError("TP bulk accept"));
    return nullptr;
  }
  auto socket = std::shared_ptr<TpBulkSocket>(new TpBulkSocket(peer));
  return Authenticate(peer, token, 0, error) ? socket : nullptr;
}

std::shared_ptr<TpBulkSocket> TpBulkSocket::Connect(const std::string& host,
                                                    std::uint16_t port,
                                                    const std::string& token,
                                                    std::string* error) {
  const auto deadline = std::chrono::steady_clock::now() + kPairTimeout;
  const std::string service = std::to_string(port);
  for (;;) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    const int resolved =
        ::getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses);
    if (resolved != 0) {
      SetError(error,
               "TP bulk getaddrinfo: " + std::string(::gai_strerror(resolved)));
      return nullptr;
    }
    int fd = -1;
    for (auto* address = addresses; address != nullptr && fd < 0;
         address = address->ai_next) {
      fd = ::socket(address->ai_family, address->ai_socktype,
                    address->ai_protocol);
      if (fd < 0) {
        continue;
      }
      (void)::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &kSocketBufferBytes,
                         sizeof(kSocketBufferBytes));
      if (::connect(fd, address->ai_addr, address->ai_addrlen) != 0) {
        ::close(fd);
        fd = -1;
      }
    }
    ::freeaddrinfo(addresses);
    if (fd >= 0) {
      auto socket = std::shared_ptr<TpBulkSocket>(new TpBulkSocket(fd));
      return Authenticate(fd, token, 1, error) ? socket : nullptr;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      SetError(error, SystemError("TP bulk connect"));
      return nullptr;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

bool TpBulkSocket::Pair(std::shared_ptr<TpBulkSocket>* rank0,
                        std::shared_ptr<TpBulkSocket>* rank1,
                        std::string* error) {
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    SetError(error, SystemError("TP bulk socketpair"));
    return false;
  }
  *rank0 = std::shared_ptr<TpBulkSocket>(new TpBulkSocket(fds[0]));
  *rank1 = std::shared_ptr<TpBulkSocket>(new TpBulkSocket(fds[1]));
  return true;
}

bool TpBulkSocket::Send(const TpBulkHeader& header,
                        std::span<const std::byte> payload,
                        std::string* error) {
  if (payload.size() != header.bytes) {
    SetError(error, "TP bulk frame size does not match its payload");
    return false;
  }
  const std::lock_guard<std::mutex> lock(send_mutex_);
  return SendAll(fd_, &header, sizeof(header), error) &&
         SendAll(fd_, payload.data(), payload.size(), error);
}

bool TpBulkSocket::ReceiveHeader(TpBulkHeader* header, std::string* error) {
  return RecvAll(fd_, header, sizeof(*header), error);
}

bool TpBulkSocket::ReceivePayload(std::span<std::byte> payload,
                                  std::string* error) {
  return RecvAll(fd_, payload.data(), payload.size(), error);
}

void TpBulkSocket::Shutdown() noexcept {
  (void)::shutdown(fd_, SHUT_RDWR);
}

}  // namespace gufo::server
