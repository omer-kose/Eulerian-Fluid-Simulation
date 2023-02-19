import taichi as ti


ti.init(ti.gpu)


x = ti.field(dtype=ti.f32, shape=(3, 5))


@ti.kernel
def test():
    for i, j in x:
        x[i, j] += 1


test()
print(x)
test()
print(x)