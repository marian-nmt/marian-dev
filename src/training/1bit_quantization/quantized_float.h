
// replace each bucket to (L + R)/2 where L and R are bucket's range
__device__ float bucketToFloat(uint8_t bucket, float step) {
  bool sign = bucket & 1;
  bucket = bucket>>1;
  if (sign)
    return step * (0.5 + bucket);
  else
    return -step * (0.5 + bucket); 
}

// get the bucket ID given a float
__device__ uint8_t floatToBucket(float input, float step, uint8_t bucket_size) {
   uint8_t bucket = min(bucket_size - 1, (int) (abs(input) / step));
   return (bucket<<1) + ((input>0)?1:0);
}

typedef union {
  uint8_t v;
  struct {
  // each float1_s contains 8 bucketed floats
  // sadly, bitfield does not support array access :(
    uint8_t bit1: 1;
    uint8_t bit2: 1;
    uint8_t bit3: 1;
    uint8_t bit4: 1;
    uint8_t bit5: 1;
    uint8_t bit6: 1;
    uint8_t bit7: 1;
    uint8_t bit8: 1;
   } bits;
  
  __device__ void fromFloat(float* inputs, float step, int b) {
    bits.bit1 = floatToBucket(inputs[0], step, b);
    bits.bit2 = floatToBucket(inputs[1], step, b);
    bits.bit3 = floatToBucket(inputs[2], step, b);
    bits.bit4 = floatToBucket(inputs[3], step, b);
    bits.bit5 = floatToBucket(inputs[4], step, b);
    bits.bit6 = floatToBucket(inputs[5], step, b);
    bits.bit7 = floatToBucket(inputs[6], step, b);
    bits.bit8 = floatToBucket(inputs[7], step, b);
  }

  __device__ void toFloat(float* outputs, float step) {
    outputs[0] = bucketToFloat(bits.bit1, step);
    outputs[1] = bucketToFloat(bits.bit2, step);
    outputs[2] = bucketToFloat(bits.bit3, step);
    outputs[3] = bucketToFloat(bits.bit4, step);
    outputs[4] = bucketToFloat(bits.bit5, step);
    outputs[5] = bucketToFloat(bits.bit6, step);
    outputs[6] = bucketToFloat(bits.bit7, step);
    outputs[7] = bucketToFloat(bits.bit8, step);
  }
} float1_s;


typedef union {
  uint8_t v;
  struct {
  // each float1_s contains 4 bucketed floats
  // sadly, bitfield does not support array access :(
    uint8_t bit1: 2;
    uint8_t bit2: 2;
    uint8_t bit3: 2;
    uint8_t bit4: 2;
   } bits;
  __device__ void fromFloat(float* inputs, float step, int b) {
    bits.bit1 = floatToBucket(inputs[0], step, b);
    bits.bit2 = floatToBucket(inputs[1], step, b);
    bits.bit3 = floatToBucket(inputs[2], step, b);
    bits.bit4 = floatToBucket(inputs[3], step, b);
  }

  __device__ void toFloat(float* outputs, float step) {
    outputs[0] = bucketToFloat(bits.bit1, step);
    outputs[1] = bucketToFloat(bits.bit2, step);
    outputs[2] = bucketToFloat(bits.bit3, step);
    outputs[3] = bucketToFloat(bits.bit4, step);
  }
} float2_s;


typedef union {
  uint8_t v;
  struct {
  // each float1_s contains 2 bucketed floats
  // sadly, bitfield does not support array access :(
    uint8_t bit1: 4;
    uint8_t bit2: 4;
   } bits;
  __device__ void fromFloat(float* inputs, float step, int b) {
    bits.bit1 = floatToBucket(inputs[0], step, b);
    bits.bit2 = floatToBucket(inputs[1], step, b);
  }

  __device__ void toFloat(float* outputs, float step) {
    outputs[0] = bucketToFloat(bits.bit1, step);
    outputs[1] = bucketToFloat(bits.bit2, step);
  }
} float4_s;

