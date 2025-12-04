struct MainPassPayload {
    vec4 color;
    float t;
};

struct TPayload {
    float t;
};

#define FLT_MAX 3.402823466e+38
#define EPSILON 0.0001
const uint AO_SPP = 1;
const float PI = 3.14159265358979323;
