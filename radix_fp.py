import struct
def radix_sort(input):
    # Convert floats to integers, modifying the sign bit
    for i in range(len(input)):
        #正数则符号位变为1，负数则取反
        input[i] = ((struct.unpack('I', struct.pack('f', input[i]))[0]) >> 31 & 0x1) and (0xFFFFFFFF - struct.unpack('I', struct.pack('f', input[i]))[0]) or (0x80000000 | struct.unpack('I', struct.pack('f', input[i]))[0])
    
    bucket = [[] for _ in range(256)]
    
    for i in range(4):
        for j in range(len(input)):
            bucket[(struct.unpack('I', struct.pack('f', input[j]))[0]) >> (i * 8) & 0xFF].append(input[j])
        
        count = 0
        for j in range(256):
            for k in range(len(bucket[j])):
                input[count] = bucket[j][k]
                count += 1
            bucket[j] = []
    
    # Convert integers back to floats
    for i in range(len(input)):
        input[i] = ((struct.unpack('I', struct.pack('f', input[i]))[0]) >> 31 & 0x1) and (struct.unpack('I', struct.pack('f', input[i]))[0] & 0x7FFFFFFF) or (0xFFFFFFFF - struct.unpack('I', struct.pack('f', input[i]))[0])

a = 3.14159
b = struct.pack('f', a)
c = struct.unpack('I', b)
print(a, b, c)