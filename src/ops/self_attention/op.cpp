#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "Self-attention tensors must be 3D");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0] && attn_val->shape()[1] == q->shape()[1],
                   "Self-attention output sequence and head dimensions must match query");
    CHECK_ARGUMENT(attn_val->shape()[2] == v->shape()[2],
                   "Self-attention output width must match value width");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0] && k->shape()[1] == v->shape()[1],
                   "Self-attention key and value sequence/head dimensions must match");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2],
                   "Self-attention query and key head dimensions must match");
    CHECK_ARGUMENT(k->shape()[0] >= q->shape()[0],
                   "Self-attention KV length must be at least the query length");
    CHECK_ARGUMENT(k->shape()[1] > 0 && q->shape()[1] % k->shape()[1] == 0,
                   "Self-attention query heads must be divisible by KV heads");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Self-attention: all tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(),
                                   q->data(),
                                   k->data(),
                                   v->data(),
                                   attn_val->dtype(),
                                   q->shape()[0],
                                   k->shape()[0],
                                   q->shape()[1],
                                   k->shape()[1],
                                   q->shape()[2],
                                   v->shape()[2],
                                   scale);
    }

    core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(),
                                   q->data(),
                                   k->data(),
                                   v->data(),
                                   attn_val->dtype(),
                                   q->shape()[0],
                                   k->shape()[0],
                                   q->shape()[1],
                                   k->shape()[1],
                                   q->shape()[2],
                                   v->shape()[2],
                                   scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
