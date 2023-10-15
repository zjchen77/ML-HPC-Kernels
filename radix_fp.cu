void radix_sort(float input[], int length)
{
    // positive: sign bit set 0 to 1
    // negative: all bit trans
    // 32bits split into 4 bytes, sort 1 byte 1 time
    for (int i=0; i<length; i++)
        reinterpret_cast<int&>(input[i]) = (reinterpret_cast<int&>(input[i])>>31 & 0x1)? ~reinterpret_cast<int&>(input[i]) : reinterpret_cast<int&>(input[i]) | 0x80000000;
    vector<float> bucket[256];
    for (int i=0; i<4; i++) {
        for (int j=0; j<length; j++)
            bucket[reinterpret_cast<int&>(input[j])>>(i*8) & 0xff].push_back(input[j]);
        int count = 0;
        for (int j=0; j<256; j++) {
            for (int k=0; k<bucket[j].size(); k++)
                input[count++] = bucket[j][k];
            bucket[j].clear();
        }
    }
    // after sort, recover
    for (int i=0; i<length; i++)
        reinterpret_cast<int&>(input[i]) = (reinterpret_cast<int&>(input[i])>>31 & 0x1)? reinterpret_cast<int&>(input[i]) & 0x7fffffff : ~reinterpret_cast<int&>(input[i]);
}