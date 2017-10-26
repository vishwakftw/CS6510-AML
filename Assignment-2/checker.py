# Activation function
def act(layer):
    return [1 if l >= 0 else 0 for l in layer]

# Part A
w1, w2, b1 = -0.5, -0.5, 0
w3, w4, b2 = -1.5, 0.5, -0.5
w5, w6, b3 = 0.5, -1.5, -0.5
w7, w8, b4 = 0.5, 0.5, -1

def inp_to_hid(x1, x2):
    h1 = w1*x1 + w2*x2 + b1
    h2 = w3*x1 + w4*x2 + b2
    h3 = w5*x1 + w6*x2 + b3
    h4 = w7*x1 + w8*x2 + b4
    return [h1, h2, h3, h4]

w_1, w_2, w_3, w_4, b_1 = 0, 1, 1, 0, -1

def hid_to_out(h1, h2, h3, h4):
    return [w_1*h1 + w_2*h2 + w_3*h3 + w_4*h4 + b_1]

print(act(inp_to_hid(0, 0)))                        # Should print [1, 0, 0, 0] h1 for (0, 0)
print(act(inp_to_hid(0, 1)))                        # Should print [0, 1, 0, 0] h2 for (0, 1)
print(act(inp_to_hid(1, 0)))                        # Should print [0, 0, 1, 0] h3 for (1, 0)
print(act(inp_to_hid(1, 1)))                        # Should print [0, 0, 0, 1] h4 for (1, 1)

print(act(hid_to_out(*act(inp_to_hid(0, 0)))))      # Should print 0
print(act(hid_to_out(*act(inp_to_hid(0, 1)))))      # Should print 1
print(act(hid_to_out(*act(inp_to_hid(1, 0)))))      # Should print 1
print(act(hid_to_out(*act(inp_to_hid(1, 1)))))      # Should print 0

# Part B
w1, w2, b1 = 1, -1, -1
w3, w4, b2 = -1, 1, -1
def inp_to_hid(x1, x2):
    h1 = w1*x1 + w2*x2 + b1
    h2 = w3*x1 + w4*x2 + b2
    return [h1, h2]

w_1, w_2, b_1 = 1, 1, -1

def hid_to_out(h1, h2):
    return [w_1*h1 + w_2*h2 + b_1]

print(act(hid_to_out(*act(inp_to_hid(0, 0)))))      # Should print 0
print(act(hid_to_out(*act(inp_to_hid(0, 1)))))      # Should print 1
print(act(hid_to_out(*act(inp_to_hid(1, 0)))))      # Should print 1
print(act(hid_to_out(*act(inp_to_hid(1, 1)))))      # Should print 0
