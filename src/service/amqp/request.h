// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once
#include <string>
#include "marian.h"
#include <amqpcpp.h>
#include <amqpcpp/libev.h>
#include <ev.h>
#include <ev++.h>

#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"

#include "service/common/translation_job.h"
#include "service/common/translation_service.h"
#include "service/api/json_request_handler.h"

namespace marian {
namespace amqp {

template<class Search>
class Request {
public:
  typedef server::TranslationService<Search> tservice_t;
  typedef server::JsonRequestHandlerBaseClass<tservice_t> reqhandler_t;
private:
  // AMQP MetaData
  std::string exchange_; // Which exchange should I respond to?
  std::string reply_to_; // Which queue should the response be posted to?
  std::string correlation_id_;
  uint64_t delivery_tag_{0};

  // Request Data
  Ptr<std::string> body_; // body of the amqp message

  // Actual request handler (templated)
  Ptr<reqhandler_t const> process_; // parses and processes body_
  Ptr<rapidjson::Document> response_;
public:
  Request() { }

  Request(Request const& other)
    : exchange_(other.exchange_),
      reply_to_(other.reply_to_),
      correlation_id_(other.correlation_id_),
      delivery_tag_(other.delivery_tag_),
      body_(other.body_),
      process_(other.process_)
  { }

  Request(uint64_t deliveryTag, const AMQP::Message& msg,
          Ptr<reqhandler_t const> processor)
    : exchange_(msg.exchange()), reply_to_(msg.replyTo()),
      correlation_id_(msg.correlationID()),
      delivery_tag_(deliveryTag),
      process_(processor) {
    body_.reset(new std::string(msg.body()));
  }

  void
  process() {
    response_ = (*process_)(*body_);
  }

  rapidjson::Document const&
  doc() const {
    return *response_;
  }

  uint64_t
  deliveryTag() const {
    return delivery_tag_;
  }

  std::string const&
  exchange() const {
    return exchange_;
  }

  std::string const&
  replyTo() const {
    return reply_to_;
  }

  std::string const&
  correlationID() {
    return correlation_id_;
  }
};

}} // end of namespace marian::amqp
