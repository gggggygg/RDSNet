import torch
import os

def test_my_deform_conv2d():
    # 模拟输入
    x = torch.randn(2, 3, 10, 10, device='cuda')
    w = torch.randn(6, 3, 3, 3, device='cuda')
    offset = torch.randn(2, 2 * 3 * 3, 8, 8, device='cuda')
    mask = torch.sigmoid(torch.randn(2, 3 * 3, 8, 8, device='cuda'))
    bias = torch.randn(6, device='cuda')



    # 参数
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 0, 0
    dil_h, dil_w = 1, 1
    groups = 1
    offset_groups = 1
    use_mask = True

    # 🔥 调用你注册的自定义 op
    y = torch.ops.my_ops.deform_conv2d(
        x, w, offset, mask, bias,
        stride_h, stride_w, pad_h, pad_w,
        dil_h, dil_w, groups, offset_groups, use_mask
    )

    print("Output shape:", y.shape)

    loss = y.sum()
    loss.backward()

    print("x.grad:", x.grad.shape if x.grad is not None else None)
    print("w.grad:", w.grad.shape if w.grad is not None else None)
    print("bias.grad:", bias.grad.shape if bias.grad is not None else None)

def test_my_deform_conv2d_backward():
    import torch
    import RDSConv_ext

    x = torch.zeros(2, 3, 10, 10, device='cuda', requires_grad=True)
    w = torch.randn(6, 3, 3, 3, device='cuda', requires_grad=True)
    offset = torch.randn(2, 2 * 3 * 3, 10, 10, device='cuda')  # 默认不反传
    mask = None
    bias = None

    mapping = torch.zeros(1, dtype=torch.int32, device='cuda')


    from RDSConv import RDSConv2d

    y = RDSConv2d(x,
                  offset=offset,  # 每像素卷积核偏置
                  weight=w,
                  mapping_idx=mapping,
                  bias=None,
                  stride=(1,1),
                  padding=(1,1),
                  mask=mask, )

    loss = y.sum()
    loss.backward()


if __name__ == "__main__":

    import torch
    import RDSConv_ext

    print(torch.ops.my_ops)
    print(dir(torch.ops.my_ops))  # 打印可用的函数名列表

    # 测试调用
    try:
        op = torch.ops.my_ops.deform_conv2d
        print("✅ deform_conv2d 已注册")

        test_my_deform_conv2d_backward()

    except Exception as e:
        print("❌ 没有注册:", e)

