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

#include "server/translation_job.h"
#include "server/translation_service.h"

namespace marian {
namespace amqp {

template<class Search=BeamSearch>
class Request {
  // AMQP::Message response_; // response message
  // AMQP::Envelope env_;

  std::string exchange_;   // Which exchange should I respond to?
  std::string reply_to_; // Which queue should the response be posted to?
  std::string correlation_id_;
  uint64_t delivery_tag_{0};
  Ptr<rapidjson::Document> request_; // the parsed payload of the amqp message
  Ptr<server::TranslationService<Search>> service_;
public:
  Request() { }

  Request(Request const& other)
    : exchange_(other.exchange_),
      reply_to_(other.reply_to_),
      correlation_id_(other.correlation_id_),
      delivery_tag_(other.delivery_tag_),
      request_(other.request_),
      service_(other.service_)
  { }

  Request(uint64_t deliveryTag, const AMQP::Message& msg,
          Ptr<server::TranslationService<Search>> service)
    : exchange_(msg.exchange()), reply_to_(msg.replyTo()),
      correlation_id_(msg.correlationID()),
      delivery_tag_(deliveryTag), service_(service) {
    request_.reset(new rapidjson::Document);
    request_->Parse(msg.body(), msg.bodySize());
  }

  void
  process() {
    server::NodeTranslation<> job(request_.get(), *service_);
    job.finish(request_->GetAllocator());
  }

  rapidjson::Document const&
  doc() const {
    return *request_;
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
