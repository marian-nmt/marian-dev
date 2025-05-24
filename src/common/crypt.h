#pragma once

#include "common/logging.h"

#include <string>
#include <stdexcept>
#include <cstring>
#include <fstream>

#include <openssl/evp.h>
#include <openssl/rand.h>

namespace marian {
namespace crypt {

// this a replacement for std::random_device that makes use of OpenSSL's RAND_bytes,
// a cryptographically secure random number generator
class OpenSSLRNG {
public:
  using result_type = std::uint32_t;

  static constexpr result_type min() {
    return std::numeric_limits<result_type>::min();
  }

  static constexpr result_type max() {
      return std::numeric_limits<result_type>::max();
  }

  result_type operator()() {
    result_type random_value;
    if (RAND_bytes(reinterpret_cast<unsigned char*>(&random_value), sizeof(random_value)) != 1) {
        ABORT("Failed to generate random number using OpenSSL RAND_bytes");
    }
    return random_value;
  }
};

// compute sha256 hash of a string, 32 bytes
inline static std::string sha256(const std::string& data) {
  unsigned char hash[EVP_MAX_MD_SIZE]; // 64 bytes
  unsigned int hash_len; // length of sha32 will be 32 bytes

  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  if (!ctx) {
    ABORT("Failed to create message digest context.");
  }

  if (1 != EVP_DigestInit_ex(ctx, EVP_sha256(), NULL)) {
    EVP_MD_CTX_free(ctx);
    ABORT("Failed to initialize message digest.");
  }

  if (1 != EVP_DigestUpdate(ctx, data.data(), data.size())) {
    EVP_MD_CTX_free(ctx);
    ABORT("Failed to update message digest.");
  }

  if (1 != EVP_DigestFinal_ex(ctx, hash, &hash_len)) {
    EVP_MD_CTX_free(ctx);
    ABORT("Failed to finalize message digest.");
  }

  EVP_MD_CTX_free(ctx);

  ABORT_IF(hash_len != 32, "SHA256 hash length is not 32 bytes??");

  return std::string(reinterpret_cast<char*>(hash), hash_len);
}

// read in a file given a file name into a std::string using modern C++
inline static std::string read_file(const std::string& file_name) {
  std::ifstream file(file_name, std::ios::binary);
  if (!file) {
    ABORT("Failed to open file '{}'", file_name);
  }
  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  return content;
}

// turn a file into a sha256 hash
inline static std::string read_file_sha256(const std::string& file_name) {
  return sha256(read_file(file_name));
}

inline static std::string get_sha256_key() {
  // get key from environment variable AES_KEY
  const char* key = std::getenv("AES_KEY");
  if (key == nullptr) {
    return "";
  } else {
    LOG(info, "Using encryption key from environment variable AES_KEY");
    return sha256(std::string(key));
  }
}

inline static std::string encrypt_aes_256_gcm(const std::string& plaintext, const std::string& key) {
  // Key size for AES-256
  const size_t key_len = 32;
  // GCM mode requires a nonce/IV of 12 bytes
  const size_t iv_len = 12;
  // GCM mode generates a tag of 16 bytes
  const size_t tag_len = 16;

  if (key.size() != key_len) {
    ABORT("Key length must be 32 bytes.");
  }

  // Generate a random IV
  std::vector<unsigned char> iv(iv_len);
  if (!RAND_bytes(iv.data(), iv_len)) {
    ABORT("Failed to generate IV.");
  }

  // Output buffers
  std::vector<unsigned char> ciphertext(plaintext.size());
  std::vector<unsigned char> tag(tag_len);

  // Encryption context
  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  if (!ctx) {
    ABORT("Failed to create cipher context.");
  }

  int len;

  // Initialize encryption operation
  if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL)) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Failed to initialize encryption.");
  }

  // Set IV length
  if (1 != EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, iv_len, NULL)) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Failed to set IV length.");
  }

  // Initialize key and IV
  if (1 != EVP_EncryptInit_ex(ctx, NULL, NULL, reinterpret_cast<const unsigned char*>(key.data()), iv.data())) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Failed to initialize key and IV.");
  }

  // Provide the message to be encrypted and obtain the encrypted output
  if (1 != EVP_EncryptUpdate(ctx, ciphertext.data(), &len, reinterpret_cast<const unsigned char*>(plaintext.data()), plaintext.size())) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Failed to encrypt plaintext.");
  }

  int ciphertext_len = len;

  // Finalize encryption
  if (1 != EVP_EncryptFinal_ex(ctx, ciphertext.data() + len, &len)) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Failed to finalize encryption.");
  }

  ciphertext_len += len;

  // Get the tag
  if (1 != EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, tag_len, tag.data())) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Failed to get tag.");
  }

  // Clean up
  EVP_CIPHER_CTX_free(ctx);

  // Combine IV, ciphertext, and tag into the final output
  std::string encrypted;
  encrypted.reserve(iv.size() + ciphertext_len + tag.size());
  encrypted.append(reinterpret_cast<const char*>(iv.data()), iv.size());
  encrypted.append(reinterpret_cast<const char*>(ciphertext.data()), ciphertext_len);
  encrypted.append(reinterpret_cast<const char*>(tag.data()), tag.size());

  return encrypted;
}

inline static std::string decrypt_aes_256_gcm(const std::string& encrypted, const std::string& key) {
  // Key size for AES-256
  const size_t key_len = 32;
  // GCM mode requires a nonce/IV of 12 bytes
  const size_t iv_len = 12;
  // GCM mode generates a tag of 16 bytes
  const size_t tag_len = 16;

  if (key.size() != key_len) {
    ABORT("Key length must be 32 bytes.");
  }

  if (encrypted.size() < iv_len + tag_len) {
    ABORT("Encrypted message is too short.");
  }

  // Extract IV, ciphertext, and tag
  std::vector<unsigned char> iv(reinterpret_cast<const unsigned char*>(encrypted.data()), reinterpret_cast<const unsigned char*>(encrypted.data()) + iv_len);
  std::vector<unsigned char> tag(reinterpret_cast<const unsigned char*>(encrypted.data()) + encrypted.size() - tag_len, reinterpret_cast<const unsigned char*>(encrypted.data()) + encrypted.size());
  std::vector<unsigned char> ciphertext(reinterpret_cast<const unsigned char*>(encrypted.data()) + iv_len, reinterpret_cast<const unsigned char*>(encrypted.data()) + encrypted.size() - tag_len);

  // Output buffer for decrypted text
  std::vector<unsigned char> plaintext(ciphertext.size());

  // Decryption context
  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  if (!ctx) {
    ABORT("Failed to create cipher context.");
  }

  int len;

  // Initialize decryption operation
  if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL)) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Failed to initialize decryption.");
  }

  // Set IV length
  if (1 != EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, iv_len, NULL)) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Failed to set IV length.");
  }

  // Initialize key and IV
  if (1 != EVP_DecryptInit_ex(ctx, NULL, NULL, reinterpret_cast<const unsigned char*>(key.data()), iv.data())) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Failed to initialize key and IV.");
  }

  // Provide the message to be decrypted and obtain the plaintext output
  if (1 != EVP_DecryptUpdate(ctx, plaintext.data(), &len, ciphertext.data(), ciphertext.size())) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Failed to decrypt ciphertext.");
  }

  int plaintext_len = len;

  // Set expected tag value
  if (1 != EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, tag_len, const_cast<unsigned char*>(tag.data()))) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Failed to set tag.");
  }

  // Finalize decryption
  if (1 != EVP_DecryptFinal_ex(ctx, plaintext.data() + len, &len)) {
    EVP_CIPHER_CTX_free(ctx);
    ABORT("Decryption failed. Tag verification failed.");
  }

  plaintext_len += len;

  // Clean up
  EVP_CIPHER_CTX_free(ctx);

  // Return the decrypted message
  return std::string(reinterpret_cast<char*>(plaintext.data()), plaintext_len);
}

} // namespace crypt
} // namespace marian
