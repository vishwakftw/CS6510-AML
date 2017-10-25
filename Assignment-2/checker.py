# Activation function
def act(layer):
    return [1 if l >= 0 else 0 for l in layer]

# Part A
def inp_to_hid(x1, x2):
    h1 = -x1 + x2 - 1
    h2 = -x1 + x2
    h3 = x1 - x2
    h4 = x1 - x2 - 1
    return [h1, h2, h3, h4]

def hid_to_out(h1, h2, h3, h4):
    return [h1 + h4 - 1]

print(act(hid_to_out(*act(inp_to_hid(0, 0)))))      # Should print 0
print(act(hid_to_out(*act(inp_to_hid(0, 1)))))      # Should print 1
print(act(hid_to_out(*act(inp_to_hid(1, 0)))))      # Should print 1
print(act(hid_to_out(*act(inp_to_hid(1, 1)))))      # Should print 0

# Part B
def inp_to_hid(x1, x2):
    h1 = x1 - x2 - 1
    h2 = -x1 + x2 - 1
    return [h1, h2]

def hid_to_out(h1, h2):
    return [h1 + h2 - 1]

print(act(hid_to_out(*act(inp_to_hid(0, 0)))))      # Should print 0
print(act(hid_to_out(*act(inp_to_hid(0, 1)))))      # Should print 1
print(act(hid_to_out(*act(inp_to_hid(1, 0)))))      # Should print 1
print(act(hid_to_out(*act(inp_to_hid(1, 1)))))      # Should print 0
