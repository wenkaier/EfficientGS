#define NUM_THREAD 256
void copy_shs(
    const int P,
    const int* old_idx,
    const float* old_shs,
    const int* old_shs_degree,
    const int* old_shs_ptr,
    float* new_shs,
    const int* new_shs_degree,
    const int* new_shs_ptr
);

