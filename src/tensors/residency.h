#pragma once

namespace marian {
 
// TODO: A better approach to dispatch
enum ResidentDevice { DEVICE_CPU, DEVICE_GPU };

template <typename T> struct residency_trait;

}
