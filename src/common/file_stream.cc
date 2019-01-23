#include "common/file_stream.h"

#include <iostream>
#include <istream>
#include <streambuf>
#include <string>
#include <unistd.h>
#include <vector>

namespace marian {
namespace io {

class ReadFDBuf : public std::streambuf {
  public:
    // Does not take ownership of file.
    explicit ReadFDBuf(int fd, std::size_t buffer_size = 4096)
      : fd_(fd), mem_(buffer_size) {
      setg(End(), End(), End());
    }

  private:
    int_type underflow() {
      if (gptr() == egptr()) {
        // Reached end, refill
        ssize_t got = Read();
        if (!got) {
          // End of file
          return traits_type::eof();
        }
        setg(Begin(), Begin(), Begin() + got);
      }
      return traits_type::to_int_type(*gptr());
    }

    // If the putback goes below the buffer, try to seek backwards.
    int_type pbackfail(int c = EOF) {
      /* "It is unspecified whether the content of the controlled input
       * sequence is modified if the function succeeds and c does not match the
       * character at that position."
       * -- http://www.cplusplus.com/reference/streambuf/streambuf/pbackfail/
       */
      if (gptr() > Begin()) {
        setg(Begin(), gptr() - 1, End());
      } else {
        if (lseek(fd_, -1, SEEK_CUR) == -1) {
          return EOF;
        }
        ssize_t got = Read();
        if (!got) {
          // This happens if the file was truncated underneath us.
          return traits_type::eof();
        }
        setg(Begin(), Begin(), Begin() + got);
      }
      return traits_type::to_int_type(*gptr());
    }

    // Read some amount into [Begin(), End()), returning the amount read.
    ssize_t Read() {
      ssize_t got;
      // Loop to keep reading if EINTR happens.
      // This way the program is robust to Ctrl+Z then backgrounding.
      do {
        errno = 0;
        got =
#ifdef _MSC_VER
          _read
#else
          read
#endif
          (fd_, Begin(), End() - Begin());
      } while (got == -1 && errno == EINTR);
      if (got < 0) {
        std::cerr << "Error" << std::endl;
        abort(); // TODO
      }
      return got;
    }

    char *Begin() { return &mem_.front(); }
    char *End() { return &mem_.back() + 1; }

    int fd_;
    std::vector<char> mem_;

    ReadFDBuf(const ReadFDBuf &) = delete;
    ReadFDBuf &operator=(const ReadFDBuf &) = delete;
};

// Write to a file descriptor.
class WriteFDBuf : public std::streambuf {
  public:
    explicit WriteFDBuf(int fd, std::size_t buffer_size = 4096)
      : fd_(fd), mem_(buffer_size) {
      setp(End(), End());
    }

    ~WriteFDBuf() { sync(); }

  private:
    int_type overflow(int c = EOF) {
      if (c == EOF) {
        // Apparently overflow(EOF) means sync().
        sync();
        return c;
      }
      if (pptr() == epptr()) {
        // Out of buffer.  Write and reset.
        sync();
        setp(Begin(), End());
      }
      // Put character on the end.
      *pptr() = traits_type::to_char_type(c);
      pbump(1);
      return c;
    }

    // Write everything in the buffer to the file.
    int sync() {
      const char *from = pbase();
      const char *to = pptr();
      while (from != to) {
        from += WriteSome(from, to);
      }
      // We die on all failures.
      return 0;
    }

    // Write part of the buffer, returning the amount written.
    ssize_t WriteSome(const char *from, const char *to) {
      ssize_t put = 0;
      do {
        errno = 0;
        put =
#ifdef _MSC_VER
          _write
#else
          write
#endif
          (fd_, from, to - from);
      } while (put == -1 && errno == EINTR);
      if (put == -1) {
        // TODO
        std::cerr << "Error writing" << std::endl;
        abort();
      }
      return put;
    }

    char *Begin() { return &mem_.front(); }
    char *End() { return &mem_.back() + 1; }

    int fd_;
    std::vector<char> mem_;

    WriteFDBuf(const WriteFDBuf &) = delete;
    WriteFDBuf &operator=(const WriteFDBuf &) = delete;
};

} // namespace io
} // namespace marian
