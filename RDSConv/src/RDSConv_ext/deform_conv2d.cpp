#include "deform_conv2d.h"
#include <iostream>


#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

#include <torch/extension.h>
#include <iostream>

//namespace my_ops {
//    at::Tensor test_f() {
//        std::cout << "test_f called!" << std::endl;
//        return at::ones({2, 3});
//    }
//}
//
//TORCH_LIBRARY(my_ops, m) {
//    std::cout << "Registering my_ops test_f..." << std::endl;
//    m.def("test_f() -> Tensor");
//    m.impl("test_f", &my_ops::test_f);
//}
//
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    // 空的，也不绑定任何 Python 函数，只是为了确保这个模块被初始化
//}

//#include <torch/extension.h>
//#include <ATen/ATen.h>
//#include <iostream>
//
//at::Tensor test_f(const at::Tensor& input, int64_t stride_h) {
//    std::cout << "test_f called, stride_h = " << stride_h << std::endl;
//    return at::ones({2, 3});
//}
//
//TORCH_LIBRARY(my_ops, m) {
//    m.def("test_f(Tensor input, int stride_h) -> Tensor");
//    m.impl("test_f", test_f);
//}
//
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    // empty
//}


namespace my_ops {

at::Tensor deform_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mapping_idx,
//    const at::Tensor& mapping_wgt,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.deform_conv2d.deform_conv2d");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("my_ops::deform_conv2d", "")
                       .typed<decltype(deform_conv2d)>();

  return op.call(
      input,
      weight,
      offset,
      mapping_idx,
//      mapping_wgt,
      mask,
      bias,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      groups,
      offset_groups,
      use_mask);
}


namespace detail {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_deform_conv2d_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
//    const at::Tensor& mapping,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("my_ops::_deform_conv2d_backward", "")
          .typed<decltype(_deform_conv2d_backward)>();
  return op.call(
      grad,
      input,
      weight,
      offset,
//      mapping,
      mask,
      bias,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      groups,
      offset_groups,
      use_mask);
}

} // namespace detail



} // namespace my_ops


TORCH_LIBRARY(my_ops, m) {
  m.def("deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor mapping_idx, Tensor mask, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> Tensor");
//  m.def("_deform_conv2d_backward(Tensor grad, Tensor input, Tensor weight, Tensor offset, Tensor mapping, Tensor mask, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("_deform_conv2d_backward(Tensor grad, Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.impl("deform_conv2d", &my_ops::deform_conv2d);
  m.impl("_deform_conv2d_backward", &my_ops::detail::_deform_conv2d_backward);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 空的，也不绑定任何 Python 函数，只是为了确保这个模块被初始化
}
