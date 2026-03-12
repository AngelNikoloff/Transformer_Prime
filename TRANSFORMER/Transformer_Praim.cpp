// === Transformer Prime ===

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <numeric>
#include <tuple>
#include <fstream>
#include <cstdint>
#include <sstream>
#include <map>
#include <set>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <omp.h>
#include <windows.h>

#pragma warning(disable : 4267)

namespace nm51 {
#define avx

    typedef float atype;
    typedef std::vector<atype> avec;
    typedef std::vector<int>   ivec;
    typedef std::vector<std::vector<atype>> _amat; // not in use
    typedef std::vector<std::vector<int>>   _imat; // not in use

    struct fmat
    {
        //  fmat  flat row-major 2-D matrix
        int rows = 0, cols = 0;
        std::vector<atype> data;

        fmat() {}
        fmat(int r, int c, atype fill = 0.0) : rows(r), cols(c), data(r* c, fill) {}

        inline atype& at(int r, int c) { return data[r * cols + c]; }
        inline atype  at(int r, int c) const { return data[r * cols + c]; }
        inline atype* row_ptr(int r) { return data.data() + r * cols; }
        inline const atype* row_ptr(int r) const { return data.data() + r * cols; }
        void zero() { std::fill(data.begin(), data.end(), 0.0); }
        void resize(int r, int c, atype fill = 0.0) { rows = r; cols = c; data.assign(r * c, fill); }

        friend inline fmat operator+(const fmat& a, const fmat& b)
        {
            assert(a.rows == b.rows && a.cols == b.cols);
            fmat out(a.rows, a.cols);
            for (int i = 0; i < (int)a.data.size(); ++i) out.data[i] = a.data[i] + b.data[i];
            return out;
        }
        friend inline fmat& operator+=(fmat& a, const fmat& b)
        {
            assert(a.rows == b.rows && a.cols == b.cols);
            for (int i = 0; i < (int)a.data.size(); ++i) a.data[i] += b.data[i];
            return a;
        }
        friend inline fmat operator-(const fmat& a, const fmat& b)
        {
            assert(a.rows == b.rows && a.cols == b.cols);
            fmat out(a.rows, a.cols);
            for (int i = 0; i < (int)a.data.size(); ++i) out.data[i] = a.data[i] - b.data[i];
            return out;
        }
        friend inline fmat operator*(const fmat& a, const fmat& b)
        {
            assert(a.rows == b.rows && a.cols == b.cols);
            fmat out(a.rows, a.cols);
            for (int i = 0; i < (int)a.data.size(); ++i) out.data[i] = a.data[i] * b.data[i];
            return out;
        }
    };

    void print_attn_heat(const fmat& A)
    {
        for (int i = 0; i < A.rows; i++) // tokens
        {
            for (int j = 0; j < A.cols; j++) // previous tokens to current token;
            {
                float v = A.at(i, j);

                char c = '.';
                if (v > 0.5) c = '5';
                else if (v > 0.2) c = '2';
                else if (v > 0.1) c = '1';

                printf("%c ", c);
            }
            printf("\n");
        }
        printf("\n");
    }

    struct ConfigMinimal
    {
        int vocabulary_size = 27;
        int dim = 32;
        int layers = 2;// 2;
        int heads = 1;
        int seq_length = 16; // 16; // block_size =  how long each context sequence is, also known as T;

        int batch_size = 8;// 8;

        float dropout_AttHead = 0.0; // 0.1 
        float dropout_AttMHA = 0.0;  // 0.1
        float dropout_FFN = 0.0;

        atype lr = 0.001;
        atype adamw_beta1 = 0.9;
        atype adamw_beta2 = 0.999;
        atype adamw_eps = 1e-8;
        atype weight_decay = 0.01;

        float temperature = 1.0f;

        int epochs = 100000;
        int steps = 20;
        int num_threads = 0;        // 0 = use all hardware threads, N = cap at N threads

        int prompt_size = 10;
        int generated_text_size = 500;

        bool synthetic = false;
        bool train = true;

    } con1;
    struct ConfigMedium
    {
        int    vocabulary_size = 27;
        int    dim = 128;
        int    layers = 4;
        int    heads = 4;
        int    seq_length = 64;

        int    batch_size = 12;// 12;

        float dropout_AttHead = 0.0; // 0.1 
        float dropout_AttMHA = 0.0;  // 0.1
        float dropout_FFN = 0.0;

        atype lr = 0.001;
        atype adamw_beta1 = 0.9;
        atype adamw_beta2 = 0.999;
        atype adamw_eps = 1e-8;
        atype weight_decay = 0.01;

        float temperature = 1.0f;

        int    epochs = 100000;
        int    steps = 1;
        int    num_threads = 0;// 0 = use all hardware threads, N = cap at N threads

        int prompt_size = 10;
        int generated_text_size = 500;

        bool synthetic = false;
        bool train = true;
    } con;
    struct ConfigBig1
    {
        // Epoch 2320/1000000  loss=0.1579
        int    vocabulary_size = 27;
        int    dim = 128;
        int    layers = 6;// 4 *********
        int    heads = 4;
        int    seq_length = 128; // 64 **************

        int    batch_size = 16;// 12;

        float dropout_AttHead = 0.0; // 0.1 
        float dropout_AttMHA = 0.0;  // 0.1
        float dropout_FFN = 0.0;

        atype lr = 0.001;
        atype adamw_beta1 = 0.9;
        atype adamw_beta2 = 0.999;
        atype adamw_eps = 1e-8;
        atype weight_decay = 0.01;

        float temperature = 1.0f;

        int    epochs = 1000000;
        int    steps = 1;
        int    num_threads = 0;// 0 = use all hardware threads, N = cap at N threads

        int prompt_size = 10;
        int generated_text_size = 500;

        bool synthetic = false;
        bool train = true;
    } con3;
    struct ConfigForSyntheticTests
    {
        int    vocabulary_size = 27;

        int    dim = 128;
        int    layers = 2;
        int    heads = 4; // 2
        int    seq_length = 32;

        int    batch_size = 12; // 32

        float dropout_AttHead = 0.0; // 0.1 
        float dropout_AttMHA = 0.0;  // 0.1
        float dropout_FFN = 0.0;

        atype lr = 0.001;
        atype adamw_beta1 = 0.9;
        atype adamw_beta2 = 0.999;
        atype adamw_eps = 1e-8;
        atype weight_decay = 0.01;

        float temperature = 1.0f;

        int    epochs = 1000000;
        int    steps = 1;
        int    num_threads = 0;// 0 = use all hardware threads, N = cap at N threads

        int prompt_size = 10;
        int generated_text_size = 500;

        bool synthetic = false;
        bool train = true;

    } con4;

    inline avec Relu(const avec& v, atype shift = 0.0)
    {
        avec r(v.size());
        for (size_t i = 0; i < v.size(); ++i) r[i] = (v[i] > shift) ? v[i] - shift : 0.0;
        return r;
    }
    inline avec Relu_derivative(const avec& v, atype shift = 0.0)
    {
        avec r(v.size());
        for (size_t i = 0; i < v.size(); ++i) r[i] = (v[i] > shift) ? 1.0 : 0.0;
        return r;
    }
    inline avec softmax(const avec& x)
    {
        atype mx = *std::max_element(x.begin(), x.end());
        avec out(x.size()); atype s = 0;
        for (size_t i = 0; i < x.size(); i++) { out[i] = std::exp(x[i] - mx); s += out[i]; }
        for (size_t i = 0; i < x.size(); i++) out[i] /= s;
        return out;
    }
    inline atype cross_entropy_loss(const fmat& logits, const ivec& targets)
    {
        atype loss = 0.0; int T = logits.rows;

        if (con.synthetic)
        {
            int t = T - 1;
            avec row(logits.row_ptr(t), logits.row_ptr(t) + logits.cols);
            avec probs = softmax(row);
            loss -= std::log(probs[targets[t]] + 1e-9);
            return loss;
        }

        for (int t = 0; t < T; t++)
        {
            avec row(logits.row_ptr(t), logits.row_ptr(t) + logits.cols);
            avec probs = softmax(row);
            loss -= std::log(probs[targets[t]] + 1e-9);
        }
        return loss / T;
    }
    inline fmat cross_entropy_grad(const fmat& logits, const ivec& targets)
    {
        int T = logits.rows, V = logits.cols;
        fmat grad(T, V, 0.0);

        if (con.synthetic)
        {
            // only last token, match loss
            int t = T - 1;
            avec row(logits.row_ptr(t), logits.row_ptr(t) + V);
            avec probs = softmax(row);
            for (int v = 0; v < V; v++) grad.at(t, v) = probs[v];  // no /T scaling
            grad.at(t, targets[t]) -= 1.0;
            return grad;
        }

        for (int t = 0; t < T; t++)
        {
            avec row(logits.row_ptr(t), logits.row_ptr(t) + V);
            avec probs = softmax(row);
            for (int v = 0; v < V; v++) grad.at(t, v) = probs[v] / T;
            grad.at(t, targets[t]) -= 1.0 / T;
        }
        return grad;
    }

    class AdamW
    {
    public:
        atype beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01;
        std::vector<atype> m, v;
        int t = 0;

        AdamW() {}
        AdamW(atype b1, atype b2, atype e, atype wd) : beta1(b1), beta2(b2), eps(e), weight_decay(wd) {}

        void step(atype* w, const atype* grad, int n, atype lr, int batch_size = 1)
        {
            if ((int)m.size() != n) { m.assign(n, 0.0); v.assign(n, 0.0); }
            ++t;
            atype scale = 1.0 / batch_size;
            atype bc1 = 1.0 - std::pow(beta1, t);
            atype bc2 = 1.0 - std::pow(beta2, t);

            for (int i = 0; i < n; ++i)
            {
                atype g = grad[i] * scale;
                m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
                atype m_hat = m[i] / bc1;
                atype v_hat = v[i] / bc2;
                w[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * w[i]);
            }
        }
    };

    class LayerNorm
    {
    public:
        avec gamma, beta, last_x_vec, gamma_grad, beta_grad;
        fmat last_x_mat;
        AdamW adam_gamma, adam_beta;

        LayerNorm() {}
        void init(int dim)
        {
            gamma.assign(dim, 1.0); beta.assign(dim, 0.0);
            gamma_grad.assign(dim, 0.0); beta_grad.assign(dim, 0.0);
            adam_gamma = AdamW(con.adamw_beta1, con.adamw_beta2, con.adamw_eps, 0.0);
            adam_beta = AdamW(con.adamw_beta1, con.adamw_beta2, con.adamw_eps, 0.0);
        }

        avec forward(const avec& x)
        {
            last_x_vec = x; int d = x.size();
            atype mean = std::accumulate(x.begin(), x.end(), 0.0) / d;
            atype var = 0.0; for (int i = 0; i < d; i++) var += (x[i] - mean) * (x[i] - mean);
            atype si = 1.0 / std::sqrt(var / d + 1e-5);
            avec out(d); for (int i = 0; i < d; i++) out[i] = gamma[i] * (x[i] - mean) * si + beta[i];
            return out;
        }

        avec backward_vec(const avec& d_out, const avec& lx)
        {
            int d = lx.size();
            atype mean = std::accumulate(lx.begin(), lx.end(), 0.0) / d;
            atype var = 0.0; for (int i = 0; i < d; i++) var += (lx[i] - mean) * (lx[i] - mean);
            atype si = 1.0 / std::sqrt(var / d + 1e-5);
            avec x_hat(d); for (int i = 0; i < d; i++) x_hat[i] = (lx[i] - mean) * si;
            for (int i = 0; i < d; i++) { gamma_grad[i] += d_out[i] * x_hat[i]; beta_grad[i] += d_out[i]; }
            avec d_xh(d); for (int i = 0; i < d; i++) d_xh[i] = d_out[i] * gamma[i];
            atype s1 = 0, s2 = 0;
            for (int i = 0; i < d; i++) { s1 += d_xh[i]; s2 += d_xh[i] * x_hat[i]; }
            avec dx(d); for (int i = 0; i < d; i++) dx[i] = si * (d_xh[i] - (s1 + x_hat[i] * s2) / d);
            return dx;
        }

        fmat forward(const fmat& x)
        {
            last_x_mat = x; fmat out(x.rows, x.cols);
            for (int t = 0; t < x.rows; ++t)
            {
                avec row(x.row_ptr(t), x.row_ptr(t) + x.cols);
                avec nr = forward(row);
                for (int c = 0; c < x.cols; ++c) out.at(t, c) = nr[c];
            }
            return out;
        }

        fmat backward(const fmat& d_out)
        {
            fmat dx(d_out.rows, d_out.cols);
            for (int t = 0; t < d_out.rows; ++t)
            {
                avec drow(d_out.row_ptr(t), d_out.row_ptr(t) + d_out.cols);
                avec lrow(last_x_mat.row_ptr(t), last_x_mat.row_ptr(t) + last_x_mat.cols);
                avec dr = backward_vec(drow, lrow);
                for (int c = 0; c < d_out.cols; ++c) dx.at(t, c) = dr[c];
            }
            return dx;
        }

        void zero_grad() { std::fill(gamma_grad.begin(), gamma_grad.end(), 0.0); std::fill(beta_grad.begin(), beta_grad.end(), 0.0); }
        void update(atype lr, int batch_size = 1)
        {
            adam_gamma.step(gamma.data(), gamma_grad.data(), gamma.size(), lr, batch_size);
            adam_beta.step(beta.data(), beta_grad.data(), beta.size(), lr, batch_size);
        }
        void reduce_from(const LayerNorm& o)
        {
            for (size_t i = 0; i < gamma_grad.size(); ++i) { gamma_grad[i] += o.gamma_grad[i]; beta_grad[i] += o.beta_grad[i]; }
        }
        void broadcast_to(LayerNorm& o) const { o.gamma = gamma; o.beta = beta; }
    };

    class Dropout
    {
    public:
        float p;
        std::vector<int> mask;
        std::mt19937 rng;
        std::uniform_real_distribution<float> dist;

        Dropout(float prob = 0.0) : p(prob), dist(0.0f, 1.0f)
        {
            //p = _dropout;
            //uint32_t seed = 1337;
            //rng.seed(seed);
        }
        void init(float _dropout = 0.0)
        {
            p = _dropout;
            dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
            std::mt19937 gen(42);
            rng.seed(gen());
        }
        void forward(fmat& x, bool train = true)
        {
            int N = x.rows * x.cols;
            mask.resize(N);

            if (!train || p == 0.0f) {
                std::fill(mask.begin(), mask.end(), 1);
                return;
            }

            for (int i = 0; i < N; ++i) {
                if (dist(rng) < p)
                {
                    x.data[i] = 0.0f;
                    mask[i] = 0;
                }
                else
                {
                    x.data[i] /= (1.0f - p);  // scale to keep expected value
                    mask[i] = 1;
                }
            }
        }
        void backward(fmat& dx)
        {
            int N = dx.rows * dx.cols;
            for (int i = 0; i < N; ++i)
            {
                dx.data[i] *= mask[i];
            }
        }

        void backward1_wrong(fmat& dx)
        {
            int N = dx.rows * dx.cols;
            for (int i = 0; i < N; ++i)
            {
                dx.data[i] *= mask[i] / (1.0f - p);  // same scale as forward
            }
        }

    };

    class FLinear
    {
        // ============================================================
        //  FLinear  –  fully-batched linear layer  [T × in] -> [T × out]
        //
        //  W stored as [out_dim × in_dim]  (transposed vs. naive layout).
        //  This makes the forward dot-product access W row-by-row (contiguous)
        //  instead of striding across columns, giving 3-4x better cache
        //  utilisation on large weight matrices like the FFN hidden layer.
        //
        //  forward : out[t][j] = b[j] + dot(W[j], x[t])     row access on W
        //  backward: dX[t][i] += sum_j dY[t][j] * W[j][i]   row access on W
        //            dW[j][i] += sum_t dY[t][j] * X[t][i]    row access on dW
        // ============================================================

        inline atype dot_avx(const atype* a, const atype* b, int n)
        {
            __m256 vsum = _mm256_setzero_ps(); // float version

            int i = 0;
            for (; i + 8 <= n; i += 8) // 8 floats per __m256
            {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);

                __m256 vm = _mm256_mul_ps(va, vb);
                vsum = _mm256_add_ps(vsum, vm);
            }

            // horizontal sum of 8 floats in vsum
            atype tmp[8];
            _mm256_storeu_ps(tmp, vsum);
            atype sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

            // remaining elements
            for (; i < n; ++i)
                sum += a[i] * b[i];

            return sum;
        }

    public:
        int in_dim = 0, out_dim = 0;
        bool use_relu = false;

        fmat W, dW;
        avec b, db;

        fmat last_x;     // [T × in_dim]
        fmat last_y;     // [T × out_dim]

        AdamW adam_W, adam_b;

        FLinear() {}
        void init(int _in, int _out, bool relu = false)
        {
            in_dim = _in; out_dim = _out; use_relu = relu;
            W.resize(out_dim, in_dim, 0.0);
            dW.resize(out_dim, in_dim, 0.0);
            b.assign(out_dim, 0.0);
            db.assign(out_dim, 0.0);

            std::mt19937 gen(42);// 42

            std::normal_distribution<atype> dist(0.0, 0.1); //000 
            //std::normal_distribution<atype> dist(0.0, 0.02); // 0.02 for mha head more good ??? ?????? // mai e po-dobre;

            for (auto& v : W.data) v = dist(gen);
            for (auto& v : b)      v = dist(gen);

            adam_W = AdamW(con.adamw_beta1, con.adamw_beta2, con.adamw_eps, con.weight_decay);
            adam_b = AdamW(con.adamw_beta1, con.adamw_beta2, con.adamw_eps, 0.0);
        }

        fmat forward(const fmat& x)
        {
            // X : [T × in_dim]  ->  Y : [T × out_dim]
            last_x = x;
            int T = x.rows;
            fmat y(T, out_dim);

            // out[t][j] = b[j] + dot(W[j], x[t])

            // #pragma omp parallel for collapse(2) schedule(static)
            for (int t = 0; t < T; ++t)
            {
                const atype* xrow = x.row_ptr(t);
                atype* yrow = y.row_ptr(t);
                for (int j = 0; j < out_dim; ++j)
                {
                    const atype* wrow = W.row_ptr(j);

                    //================ if not avx supported
#ifndef avx
                    atype s = b[j];
                    for (int i = 0; i < in_dim; ++i) s += wrow[i] * xrow[i];
#else if
//================ if avx supported
                    atype s = b[j] + dot_avx(wrow, xrow, in_dim);
#endif

                    yrow[j] = s;
                }
            }

            if (use_relu)
            {
                // #pragma omp parallel for
                for (int i = 0; i < T* out_dim; ++i) y.data[i] = y.data[i] > 0.0 ? y.data[i] : 0.0; //000

                //for (int i = 0; i < T* out_dim; ++i) y.data[i] = (y.data[i] * y.data[i]) > 0.0 ? y.data[i] : 0.0; // relu2

                last_y = y;
            }

            return y;
        }
        fmat backward(const fmat& dY)
        {
            // dY : [T × out_dim]  ->  dX : [T × in_dim]
            int T = dY.rows;
            fmat dY_eff = dY;

            //dY_eff = dY_eff * dY_eff; // relu2 experiment ********

            if (use_relu)
                for (int i = 0; i < T * out_dim; ++i)
                    if (last_y.data[i] <= 0.0) dY_eff.data[i] = 0.0;

            fmat dX(T, in_dim, 0.0);

            // dX[t][i] = sum_j dY[t][j] * W[j][i]
            // Per token row: for each j add W[j] (contiguous) scaled by dY[t][j]

            for (int t = 0; t < T; ++t)
            {
                const atype* drow = dY_eff.row_ptr(t);
                atype* dxrow = dX.row_ptr(t);
                for (int j = 0; j < out_dim; ++j)
                {
                    atype d = drow[j];
                    if (d == 0.0) continue;
                    const atype* wrow = W.row_ptr(j);
                    for (int i = 0; i < in_dim; ++i) dxrow[i] += d * wrow[i];
                }
            }

            // dW[j][i] += sum_t dY[t][j] * X[t][i]
            // Parallelise over output rows j – each thread writes its own dW[j]

            for (int j = 0; j < out_dim; ++j)
            {
                atype* dwrow = dW.row_ptr(j);
                atype dbi = 0.0;
                for (int t = 0; t < T; ++t)
                {
                    atype d = dY_eff.at(t, j);
                    if (d == 0.0) continue;
                    dbi += d;
                    const atype* xrow = last_x.row_ptr(t);
                    for (int i = 0; i < in_dim; ++i) dwrow[i] += d * xrow[i];
                }
                db[j] += dbi;
            }

            return dX;
        }

        void zero_grad() { dW.zero(); std::fill(db.begin(), db.end(), 0.0); }
        void update(atype lr, int batch_size = 1)
        {
            adam_W.step(W.data.data(), dW.data.data(), out_dim * in_dim, lr, batch_size);
            adam_b.step(b.data(), db.data(), out_dim, lr, batch_size);
        }

        void reduce_from(const FLinear& o)
        {
            for (int i = 0; i < out_dim * in_dim; ++i) dW.data[i] += o.dW.data[i];
            for (int j = 0; j < out_dim; ++j)          db[j] += o.db[j];
        }
        void broadcast_to(FLinear& o) const { o.W = W; o.b = b; }
    };

    class FeedForwardNetwork
    {
        FLinear ffn1, ffn2, ffn3;
        Dropout dropout;

    public:
        FeedForwardNetwork() {}

        void init(int dim, int out)
        {
            ffn1.init(dim, dim * 4, false);    // linear
            ffn2.init(dim * 4, dim * 4, true); // relu
            ffn3.init(dim * 4, dim, false);    // linear
            dropout.init(con.dropout_FFN);
        }

        fmat forward(fmat x)
        {
            x = ffn1.forward(x);
            x = ffn2.forward(x);
            if (con.train && con.dropout_FFN) dropout.forward(x);
            x = ffn3.forward(x);
            return x;
        }
        fmat backward(fmat e)
        {
            e = ffn3.backward(e);
            if (con.train && con.dropout_FFN) dropout.backward(e);
            e = ffn2.backward(e);
            e = ffn1.backward(e);
            return e;
        }

        void zero_grad() { ffn1.zero_grad();    ffn2.zero_grad();    ffn3.zero_grad(); }
        void update(atype lr, int bs = 1) { ffn1.update(lr, bs); ffn2.update(lr, bs); ffn3.update(lr, bs); }

        void reduce_from(const FeedForwardNetwork& o) { ffn1.reduce_from(o.ffn1);  ffn2.reduce_from(o.ffn2);  ffn3.reduce_from(o.ffn3); }
        void broadcast_to(FeedForwardNetwork& o) const { ffn1.broadcast_to(o.ffn1); ffn2.broadcast_to(o.ffn2); ffn3.broadcast_to(o.ffn3); }
    };

    class AttentionHeadRoPE    //+ Epoch 2120/100000  loss=1.6662 ald back
    {
    public:
        int embed_size = 0;
        FLinear Wq, Wk, Wv;
        fmat last_input, last_attn, last_q, last_k, last_v, last_q_pre, last_k_pre;
        Dropout dropout;
        fmat attn_post_dropout;

        AttentionHeadRoPE() {}
        void init(int _embed_size)
        {
            embed_size = _embed_size;
            Wq.init(embed_size, embed_size);
            Wk.init(embed_size, embed_size);
            Wv.init(embed_size, embed_size);
            dropout.init(con.dropout_AttHead);  // initialize with config value
        }

        void apply_rope(const atype* in, atype* out, int D, int pos)
        {
            int half = D / 2;
            for (int d = 0; d < half; ++d)
            {
                atype theta = pos / std::pow(10000.0, 2.0 * d / D);
                atype c = std::cos(theta), s = std::sin(theta);
                out[d] = in[d] * c - in[d + half] * s;
                out[d + half] = in[d] * s + in[d + half] * c;
            }
        }
        void apply_rope_bwd(const atype* dout, atype* dx, int D, int pos)
        {
            int half = D / 2;
            for (int d = 0; d < half; ++d)
            {
                atype theta = pos / std::pow(10000.0, 2.0 * d / D);
                atype c = std::cos(theta), s = std::sin(theta);
                dx[d] = dout[d] * c + dout[d + half] * s;
                dx[d + half] = -dout[d] * s + dout[d + half] * c;
            }
        }

        fmat forward(const fmat& x)
        {
            int T = x.rows, D = x.cols;
            last_input = x;
            last_q_pre = Wq.forward(x);
            last_k_pre = Wk.forward(x);
            last_v = Wv.forward(x);
            last_q.resize(T, D);
            last_k.resize(T, D);

            for (int i = 0; i < T; ++i) {
                apply_rope(last_q_pre.row_ptr(i), last_q.row_ptr(i), D, i);
                apply_rope(last_k_pre.row_ptr(i), last_k.row_ptr(i), D, i);
            }

            atype scale = 1.0 / std::sqrt((atype)D);
            last_attn.resize(T, T, 0.0);
            fmat attn_out(T, D, 0.0);

            for (int i = 0; i < T; ++i)
            {
                const atype* qi = last_q.row_ptr(i); // scale here = qi_scaled[d] = qi[d] * scale - instead here - scores[j] = dot * scale; for speed
                avec scores(T, -1e9);
                for (int j = 0; j <= i; ++j)
                {
                    atype dot = 0.0;
                    const atype* kj = last_k.row_ptr(j);
                    for (int d = 0; d < D; ++d) dot += qi[d] * kj[d];
                    scores[j] = dot * scale;
                }

                atype max_s = *std::max_element(scores.begin(), scores.begin() + i + 1);
                atype sum_e = 0.0;
                for (int j = 0; j <= i; ++j) { scores[j] = std::exp(scores[j] - max_s); sum_e += scores[j]; }
                for (int j = 0; j <= i; ++j) { scores[j] /= sum_e; last_attn.at(i, j) = scores[j]; }

            }

            attn_post_dropout = last_attn;

            if (con.dropout_AttHead && con.train) dropout.forward(attn_post_dropout);

            for (int i = 0; i < T; ++i)
            {
                atype* out_i = attn_out.row_ptr(i);
                for (int j = 0; j <= i; ++j)
                {
                    atype a = attn_post_dropout.at(i, j);

                    const atype* vj = last_v.row_ptr(j);
                    for (int d = 0; d < D; ++d) out_i[d] += a * vj[d];
                }
            }

            return attn_out;
        }
        fmat backward(const fmat& dout)
        {
            int T = last_input.rows, D = last_input.cols;
            atype scale = 1.0 / std::sqrt((atype)D);
            fmat dv(T, D, 0.0), dq(T, D, 0.0), dk(T, D, 0.0);

            // Step 1: Backward through value weighted sum, build dattn matrix (gradient w.r.t. last_attn, post-dropout)
            fmat dattn_mat(T, T, 0.0);
            fmat dv_acc(T, D, 0.0);
            for (int i = 0; i < T; ++i)
            {
                const atype* dout_i = dout.row_ptr(i);
                for (int j = 0; j <= i; ++j)
                {
                    //atype a = last_attn.at(i, j);  // already dropout-masked value
                    atype a = attn_post_dropout.at(i, j);

                    const atype* vj = last_v.row_ptr(j);
                    atype* dvj = dv_acc.row_ptr(j);
                    for (int d = 0; d < D; ++d)
                    {
                        dattn_mat.at(i, j) += dout_i[d] * vj[d];
                        dvj[d] += a * dout_i[d];
                    }
                }
            }
            dv = dv_acc;

            // Step 2: dropout backward on dattn_mat - Gates out the same positions that were zeroed in forward
            if (con.dropout_AttHead && con.train) dropout.backward(dattn_mat);

            // Step 3: backward through softmax + scaled dot product
            for (int i = 0; i < T; ++i)
            {
                // Backward through softmax - Softmax backward formula requires A, not A post dropout.
                atype dot_sum = 0.0;  for (int j = 0; j <= i; ++j) dot_sum += dattn_mat.at(i, j) * last_attn.at(i, j);
                avec ds(T, 0.0);      for (int j = 0; j <= i; ++j) ds[j] = last_attn.at(i, j) * (dattn_mat.at(i, j) - dot_sum);

                // Backward through scaled dot product
                atype* dq_i = dq.row_ptr(i);
                for (int j = 0; j <= i; ++j)
                {
                    atype dsj = ds[j] * scale;
                    const atype* kj = last_k.row_ptr(j);
                    const atype* qi = last_q.row_ptr(i);
                    atype* dk_j = dk.row_ptr(j);
                    for (int d = 0; d < D; ++d) { dq_i[d] += dsj * kj[d]; dk_j[d] += dsj * qi[d]; }
                }
            }

            fmat dq_pre(T, D, 0.0), dk_pre(T, D, 0.0);
            for (int i = 0; i < T; ++i)
            {
                apply_rope_bwd(dq.row_ptr(i), dq_pre.row_ptr(i), D, i);
                apply_rope_bwd(dk.row_ptr(i), dk_pre.row_ptr(i), D, i);
            }

            fmat dx_q = Wq.backward(dq_pre);
            fmat dx_k = Wk.backward(dk_pre);
            fmat dx_v = Wv.backward(dv);
            fmat dx(T, D, 0.0);
            for (int i = 0; i < T * D; ++i) dx.data[i] = dx_q.data[i] + dx_k.data[i] + dx_v.data[i];
            return dx;
        }

        void zero_grad() { Wq.zero_grad();            Wk.zero_grad();            Wv.zero_grad(); }
        void update(atype lr, int batch_size = 1) { Wq.update(lr, batch_size); Wk.update(lr, batch_size); Wv.update(lr, batch_size); }
        void reduce_from(const AttentionHeadRoPE& o) { Wq.reduce_from(o.Wq);      Wk.reduce_from(o.Wk);      Wv.reduce_from(o.Wv); }
        void broadcast_to(AttentionHeadRoPE& o) const { Wq.broadcast_to(o.Wq);     Wk.broadcast_to(o.Wk);     Wv.broadcast_to(o.Wv); }
    };

    class MultiHeadAttention
    {
    public:
        int embed_size = 0, num_heads = 0, head_dim = 0;
        std::vector<AttentionHeadRoPE> heads;
        FLinear Wo;
        fmat last_concat;
        Dropout dropout;

        MultiHeadAttention() {}
        void init(int _embed_size, int _num_heads = 1, atype = 1.0)
        {
            embed_size = _embed_size;
            num_heads = _num_heads;
            head_dim = embed_size / num_heads;
            heads.resize(num_heads);

            for (auto& h : heads) h.init(head_dim);
            Wo.init(embed_size, embed_size);
            dropout.init(con.dropout_AttMHA);
        }

        fmat slice_head(const fmat& x, int h) const
        {
            int T = x.rows, off = h * head_dim;
            fmat out(T, head_dim);
            for (int t = 0; t < T; ++t)
                for (int d = 0; d < head_dim; ++d) out.at(t, d) = x.at(t, off + d);
            return out;
        }
        fmat concat_heads(const std::vector<fmat>& ho) const
        {
            int T = ho[0].rows; fmat out(T, embed_size, 0.0);
            for (int h = 0; h < num_heads; ++h)
            {
                int off = h * head_dim;
                for (int t = 0; t < T; ++t)
                    for (int d = 0; d < head_dim; ++d) out.at(t, off + d) = ho[h].at(t, d);
            }
            return out;
        }

        fmat forward(const fmat& x)
        {
            std::vector<fmat> ho(num_heads);

            for (int h = 0; h < num_heads; ++h) ho[h] = heads[h].forward(slice_head(x, h));

            last_concat = concat_heads(ho);

            fmat out = Wo.forward(last_concat);

            if (con.dropout_AttMHA && con.train) dropout.forward(out);

            return out;
        }
        fmat backward(const fmat& _dout)
        {
            fmat dout = _dout;
            if (con.dropout_AttMHA && con.train) dropout.backward(dout);

            fmat dconcat = Wo.backward(dout);
            int T = dconcat.rows; fmat dx(T, embed_size, 0.0);
            std::vector<fmat> dxh(num_heads);

            for (int h = 0; h < num_heads; ++h)
            {
                int off = h * head_dim; fmat dslice(T, head_dim, 0.0);
                for (int t = 0; t < T; ++t) for (int d = 0; d < head_dim; ++d) dslice.at(t, d) = dconcat.at(t, off + d);
                dxh[h] = heads[h].backward(dslice);
            }
            for (int h = 0; h < num_heads; ++h)
            {
                int off = h * head_dim;
                for (int t = 0; t < T; ++t) for (int d = 0; d < head_dim; ++d) dx.at(t, off + d) += dxh[h].at(t, d);
            }
            return dx;
        }

        void zero_grad() { for (auto& h : heads) h.zero_grad(); Wo.zero_grad(); }
        void update(atype lr, int batch_size = 1) { for (auto& h : heads) h.update(lr, batch_size); Wo.update(lr, batch_size); }

        void reduce_from(const MultiHeadAttention& o) { for (int h = 0; h < num_heads; ++h) heads[h].reduce_from(o.heads[h]);  Wo.reduce_from(o.Wo); }
        void broadcast_to(MultiHeadAttention& o) const { for (int h = 0; h < num_heads; ++h) heads[h].broadcast_to(o.heads[h]); Wo.broadcast_to(o.Wo); }
    };

    class TBlock
    {
        int dim = -1;
        MultiHeadAttention att;
        FeedForwardNetwork ffn;
        LayerNorm norm1, norm2;
    public:
        TBlock() {}
        TBlock(int _dim) : dim(_dim) { att.init(dim, con.heads); ffn.init(dim, dim); norm1.init(dim); norm2.init(dim); }

        fmat forward(fmat x)
        {
            x = x + att.forward(norm1.forward(x));
            x = x + ffn.forward(norm2.forward(x));
            return x;
        }
        fmat backward(fmat e)
        {
            e += norm2.backward(ffn.backward(e));
            e += norm1.backward(att.backward(e));
            return e;
        }

        void zero_grad() { ffn.zero_grad();         norm1.zero_grad();       norm2.zero_grad();           att.zero_grad(); }
        void update(atype lr, int bs = 1) { ffn.update(lr, bs);      norm1.update(lr, bs);    norm2.update(lr, bs);        att.update(lr, bs); }
        void reduce_from(const TBlock& o) { att.reduce_from(o.att);  ffn.reduce_from(o.ffn);  norm1.reduce_from(o.norm1);  norm2.reduce_from(o.norm2); }
        void broadcast_to(TBlock& o) const { att.broadcast_to(o.att); ffn.broadcast_to(o.ffn); norm1.broadcast_to(o.norm1); norm2.broadcast_to(o.norm2); }
    };

    class Embedding
    {
    public:
        int dim = 0; fmat vocabulary, vocab_grad;
        ivec tokens;
        AdamW adam_emb;

        void init(int vocab_size, int _dim, int = 1)
        {
            dim = _dim; vocabulary.resize(vocab_size, dim, 0.0); vocab_grad.resize(vocab_size, dim, 0.0);
            std::mt19937 rng(42); std::uniform_real_distribution<atype> dist(-0.1, 0.1);
            for (auto& v : vocabulary.data) v = dist(rng);
            adam_emb = AdamW(con.adamw_beta1, con.adamw_beta2, con.adamw_eps, 0.0);
        }

        fmat forward(const ivec& toks)
        {
            tokens = toks;
            int T = toks.size();
            fmat out(T, dim);

            for (int t = 0; t < T; ++t) std::copy(vocabulary.row_ptr(toks[t]), vocabulary.row_ptr(toks[t]) + dim, out.row_ptr(t));

            return out;
        }
        void backward(const fmat& errors)
        {
            for (int t = 0; t < (int)tokens.size(); ++t)
            {
                atype* g = vocab_grad.row_ptr(tokens[t]);
                const atype* e = errors.row_ptr(t);
                for (int j = 0; j < dim; ++j) g[j] += e[j];
            }
        }

        void zero_grad() { vocab_grad.zero(); }
        void update(atype lr, int batch_size = 1)
        {
            adam_emb.step(vocabulary.data.data(), vocab_grad.data.data(), vocabulary.rows * vocabulary.cols, lr, batch_size);
            vocab_grad.zero();
        }
        void reduce_from(const Embedding& o) { for (int i = 0; i < (int)vocab_grad.data.size(); ++i) vocab_grad.data[i] += o.vocab_grad.data[i]; }
        void broadcast_to(Embedding& o) const { o.vocabulary = vocabulary; }
    };

    class Model
    {
    public:
        Embedding embedding; std::vector<TBlock> layers;
        FLinear  head;
        LayerNorm final_norm;
        fmat logits;

        void init()
        {
            embedding.init(con.vocabulary_size, con.dim, 1);
            layers.clear();
            for (int i = 0; i < con.layers; i++) layers.emplace_back(con.dim);
            head.init(con.dim, con.vocabulary_size, false);
            final_norm.init(con.dim);
        }

        fmat forward(const ivec& tokens)
        {
            fmat x = embedding.forward(tokens);
            for (auto& l : layers) x = l.forward(x);
            x = final_norm.forward(x);
            logits = head.forward(x);
            return logits;
        }
        void backward(const fmat& d_logits)
        {
            fmat dx = head.backward(d_logits);
            dx = final_norm.backward(dx);
            for (int i = (int)layers.size() - 1; i >= 0; i--) dx = layers[i].backward(dx);
            embedding.backward(dx);
        }

        void zero_grad()
        {
            head.zero_grad();
            final_norm.zero_grad();
            for (auto& l : layers) l.zero_grad();
            embedding.zero_grad();
        }
        void update(atype lr, int batch_size = 1)
        {
            head.update(lr, batch_size); final_norm.update(lr, batch_size);
            for (auto& l : layers) l.update(lr, batch_size);
            embedding.update(lr, batch_size);
        }

        void reduce_from(const Model& w)
        {
            // Accumulate worker gradients into master
            embedding.reduce_from(w.embedding);
            for (int i = 0; i < (int)layers.size(); ++i) layers[i].reduce_from(w.layers[i]);
            final_norm.reduce_from(w.final_norm); head.reduce_from(w.head);
        }
        void broadcast_to(Model& w) const
        {
            // Push master weights to worker
            embedding.broadcast_to(w.embedding);
            for (int i = 0; i < (int)layers.size(); ++i) layers[i].broadcast_to(w.layers[i]);
            final_norm.broadcast_to(w.final_norm); head.broadcast_to(w.head);
        }

        int predict_next_best(const ivec& ctx)
        {
            fmat lg = forward(ctx);
            const atype* last = lg.row_ptr(lg.rows - 1);
            int best = 0;

            for (int i = 1; i < lg.cols; ++i) if (last[i] > last[best]) best = i;

            return best;
        }
        int predict_next(const ivec& ctx, float temperature = 1.0f)
        {
            /*
            temperature 1.0 → baseline
            temperature 0.7 → по-структуриран текст
            temperature 1.3 → повече разнообразие, но грешки
            Ако при temperature = 0.7 текстът започва да има разпознаваеми думи и фрази → loss ~2.0 е достатъчен за “смислен текст” при твоите параметри.
            */

            // 1 Forward pass to get logits
            fmat lg = forward(ctx);
            const atype* last = lg.row_ptr(lg.rows - 1);  // last timestep

            int vocab_size = lg.cols;
            std::vector<float> temp_logits(vocab_size);

            // 2 Apply temperature
            for (int i = 0; i < vocab_size; ++i) temp_logits[i] = last[i] / temperature;

            // 3 Convert logits → probabilities (softmax)
            std::vector<float> probs(vocab_size);
            float max_logit = *std::max_element(temp_logits.begin(), temp_logits.end());
            float sum = 0.0f;
            for (int i = 0; i < vocab_size; ++i)
            {
                probs[i] = std::exp(temp_logits[i] - max_logit);  // for numerical stability
                sum += probs[i];
            }

            for (int i = 0; i < vocab_size; ++i) probs[i] /= sum;

            // 4 Sample next token from probs
            float r = static_cast<float>(rand()) / RAND_MAX;
            float accum = 0.0f;
            for (int i = 0; i < vocab_size; ++i)
            {
                accum += probs[i];
                if (r < accum) return i;
            }

            return vocab_size - 1; // fallback
        }
    };

    class DataUTF8
    {
    public:
        std::string txt;
        ivec corpus;
        std::vector<std::string> chars;
        std::map<std::string, int> char2idx;
        std::map<int, std::string> idx2char;

        std::vector<std::string> utf8_split(const std::string& s)
        {
            std::vector<std::string> out;

            for (size_t i = 0; i < s.size(); )
            {
                unsigned char c = s[i];
                size_t len = 1;

                if ((c & 0x80) == 0) len = 1;
                else if ((c & 0xE0) == 0xC0) len = 2;
                else if ((c & 0xF0) == 0xE0) len = 3;
                else if ((c & 0xF8) == 0xF0) len = 4;

                out.push_back(s.substr(i, len));
                i += len;
            }

            return out;
        }

        void loadFromFile(const std::string& path)
        {
            std::ifstream f(path);
            if (f.good())
            {
                txt = std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                std::cout << "Loaded: " << path << " (" << txt.size() << " chars)\n";
            }
            else
            {
                std::cout << "No file. Using built-in text.\n";
                txt = "hello world this is a simple transformer test the model learns character level language modeling ";
                std::string base = txt; for (int i = 0; i < 10; i++) txt += base;
            }
        }
        void prepare()
        {
            if (con.synthetic) return prepare_Synthetic();

            auto tokens = utf8_split(txt);

            /*
            if (con.synthetic)
            for (int i = 0; i < tokens.size(); i++)
            {
                if (tokens[i] == "\n")
                {
                    con.seq_length = i;
                    con.prompt_size = con.seq_length - 1;
                    con.generated_text_size = 1;
                    std::cout << "i: " << i << " con.seq_length: " << con.seq_length << std::endl;
                    break;
                }
            }
            */


            std::set<std::string> cs(tokens.begin(), tokens.end());
            chars = std::vector<std::string>(cs.begin(), cs.end());

            con.vocabulary_size = chars.size();

            for (size_t i = 0; i < chars.size(); i++)
            {
                char2idx[chars[i]] = i;
                idx2char[i] = chars[i];
            }

            corpus.clear();

            for (auto& t : tokens) corpus.push_back(char2idx[t]);


            std::cout << "Vocab: " << con.vocabulary_size << "\nCorpus: " << corpus.size() << "\n";

            std::cout << "_";
            for (int i = 0; i < con.vocabulary_size; i++) std::cout << idx2char[i] << "_";// << here not printed correctly
            std::cout << "\n\n";
        }
        std::tuple<ivec, ivec> getBatch(std::mt19937& rng)
        {
            if (con.synthetic) return getBatch_Synthetic(rng);

            // Thread-safe: each caller passes its own rng
            int max_start = (int)corpus.size() - con.seq_length - 1;
            std::uniform_int_distribution<int> dist(0, max_start);
            int s = dist(rng);
            return { ivec(corpus.begin() + s, corpus.begin() + s + con.seq_length),
                     ivec(corpus.begin() + s + 1, corpus.begin() + s + con.seq_length + 1) };
        }

        // ++ but only for const tasks size; if use than switch off prepare_Synthetic in prepare;
        std::tuple<ivec, ivec> getBatch_Synthetic0(std::mt19937& rng)
        {
            int seq_length = con.seq_length;
            int first_row_size = con.seq_length + 1;// for /n

            int num_rows = corpus.size() / first_row_size;

            std::uniform_int_distribution<int> dist(0, num_rows - 1);
            int start = dist(rng) * first_row_size;

            ivec x(seq_length - 1);
            ivec y(seq_length - 1);

            for (int i = 0; i < seq_length - 1; i++)
            {
                x[i] = corpus[start + i];
                y[i] = corpus[start + i + 1];
            }

            //std::cout.flush();
            //for (int t : x) std::cout << idx2char[t]; std::cout << " "; 
            //for (int t : y) std::cout << idx2char[t]; std::cout << std::endl; std::cout.flush();

            return { x, y };
        }

        //=== Synthetic
        std::vector<ivec> lines;
        void printLines(int n = 10) const
        {
            int count = n;
            //int count = std::min(n, (int)lines.size());

            for (int r = 0; r < count; r++)
            {
                for (int c = 0; c < (int)lines[r].size(); c++)  std::cout << idx2char.at(lines[r][c]);
                std::cout << "\n";
            }
        }
        void prepare_Synthetic()
        {
            // --- Build vocab from ALL characters (excluding '\n') ---
            auto all_tokens = utf8_split(txt);
            std::set<std::string> cs;
            for (auto& t : all_tokens)
                if (t != "\n") cs.insert(t);

            chars = std::vector<std::string>(cs.begin(), cs.end());
            con.vocabulary_size = chars.size();

            char2idx.clear(); idx2char.clear();
            for (size_t i = 0; i < chars.size(); i++)
            {
                char2idx[chars[i]] = (int)i;
                idx2char[(int)i] = chars[i];
            }

            // --- Fill 2D lines vector, one row per line ---
            lines.clear();
            std::istringstream ss(txt);
            std::string line;
            while (std::getline(ss, line))
            {
                if (line.empty()) continue;
                auto toks = utf8_split(line);
                ivec encoded;
                for (auto& t : toks)
                {
                    auto it = char2idx.find(t);
                    if (it != char2idx.end())
                        encoded.push_back(it->second);
                }
                if (!encoded.empty())
                    lines.push_back(encoded);
            }

            con.seq_length = lines[0].size(); // all lines must be equal size
            con.prompt_size = con.seq_length - 1;
            con.generated_text_size = 1;

            std::cout << "Vocab size : " << con.vocabulary_size << "\n";
            std::cout << "Lines      : " << lines.size() << "\n";
            std::cout << "Seq length : " << con.seq_length << "\n";

            std::cout << "_";
            for (int i = 0; i < con.vocabulary_size; i++) std::cout << idx2char[i] << "_";// << here not printed correctly
            std::cout << "\n\n";

            printLines();
        }
        std::tuple<ivec, ivec> getBatch_Synthetic(std::mt19937& rng)
        {
            std::uniform_int_distribution<int> dist(0, (int)lines.size() - 1);
            const ivec& row = lines[dist(rng)];

            ivec x(row.begin(), row.end() - 1);
            ivec y(row.begin() + 1, row.end());
            return { x, y };
        }

    };

    class Pretrain
    {
    public:
        DataUTF8  data;
        Model trans;

        void train()
        {
            std::cout << "\n=== Training..." << "\n";

            con.train = true;

            // Only one level of OpenMP parallelism. The outer batch loop owns ALL threads.
            // One Model replica per thread.
            // Workers hold weights + activation cache + gradients.
            // The master (trans) holds AdamW moments and is never touched inside the parallel region.

            int nthreads = (con.num_threads > 0) ? con.num_threads : omp_get_max_threads();
            omp_set_num_threads(nthreads);
            omp_set_nested(0);

            std::vector<Model> workers(nthreads);
            for (auto& w : workers) w.init();

            // Per-thread RNGs – different seeds so each thread samples different sequences
            std::vector<std::mt19937> rngs(nthreads);
            for (int i = 0; i < nthreads; ++i) rngs[i].seed(123 + i * 17);

            auto t0 = std::chrono::steady_clock::now();

            for (int epoch = 0; epoch < con.epochs; epoch++)
            {
                auto te = std::chrono::steady_clock::now();
                atype total_loss = 0.0;

                for (int step = 0; step < con.steps; step++)
                {
                    // ── 1. push master weights to all worker replicas (read-only during parallel section; only weights, not grads)
                    for (int tid = 0; tid < nthreads; ++tid) trans.broadcast_to(workers[tid]);
                    for (auto& w : workers) w.zero_grad();

                    atype step_loss = 0.0;
                    int B = con.batch_size;

                    // ── 2. parallel forward + backward over the batch,  
                    // Each thread i handles samples { i, i+nthreads, i+2*nthreads, … }
                    // Workers are independent – no shared mutable state in this region.
#pragma omp parallel for schedule(static) reduction(+:step_loss) num_threads(nthreads)
                    for (int b = 0; b < B; b++)
                    {
                        int tid = omp_get_thread_num();
                        ivec inp, tgt;
                        std::tie(inp, tgt) = data.getBatch(rngs[tid]);
                        fmat lg = workers[tid].forward(inp);
                        step_loss += cross_entropy_loss(lg, tgt);
                        workers[tid].backward(cross_entropy_grad(lg, tgt));
                    }

                    // ── 3. reduce gradients from all workers into master, Serial – fast (memory bandwidth limited, ~2ms per worker)
                    trans.zero_grad();
                    for (int tid = 0; tid < nthreads; ++tid) trans.reduce_from(workers[tid]);

                    // ── 4. single AdamW step on master
                    trans.update(con.lr, B);
                    total_loss += step_loss / B;
                }

                auto   now = std::chrono::steady_clock::now();
                double ep_sec = std::chrono::duration<double>(now - te).count(); //auto ep_sec = std::chrono::duration_cast<std::chrono::milliseconds>(now - te).count();
                double elapsed = std::chrono::duration<double>(now - t0).count();
                double eta = ep_sec * (con.epochs - epoch - 1);

                if ((epoch + 1) % 10 == 0 || epoch == 0)
                {
                    std::cout
                        << "Epoch " << std::setw(4) << epoch + 1 << "/" << con.epochs
                        << "  loss=" << std::fixed << std::setprecision(4) << total_loss / con.steps
                        << "  epoch_time=" << std::setprecision(2) << ep_sec << "s"
                        << "  elapsed=" << elapsed << "s  ETA=" << eta << "s\n";
                }

                if (GetAsyncKeyState('T') & 0x0001)
                {
                    con.train = false;
                    test();
                    con.train = true;
                }
                if (GetAsyncKeyState('F') & 0x0001)
                {
                    std::cout << "\n[F] pressed - stopping training...\n";
                    break;
                }
            }

            std::cout << "Done. Total: " << std::chrono::duration<atype>(std::chrono::steady_clock::now() - t0).count() << "s\n";
        }
        void test()
        {
            con.train = false;

            std::cout << "\n========== Generation ==========\n\n";

            if (con.synthetic == false)
                if ((int)data.corpus.size() < con.seq_length + 1) { std::cout << "Not enough data.\n"; return; }

            if (con.synthetic)
            {
                data.corpus = data.lines[0];
                con.prompt_size = data.corpus.size() - 1;
            }


            ivec prompt(data.corpus.begin(), data.corpus.begin() + con.prompt_size);

            std::cout << "Prompt: \n\"";
            for (int t : prompt) std::cout << data.idx2char[t];


            std::cout << "\"\n\n========== Generated - No Temperature - debug purpose: \n\n\"";
            for (int t : prompt) std::cout << data.idx2char[t];  //  std::cout << "/";
            ivec ctx = prompt;
            for (int i = 0; i < con.generated_text_size; i++)
            {
                int next = trans.predict_next_best(ctx);
                std::cout << data.idx2char[next];
                std::cout.flush();
                ctx.erase(ctx.begin());
                ctx.push_back(next);
            }
            std::cout << "\"\n\n==========";


            std::cout << "\"\n\n========== Generated with Temperature!!!: \n\n\"";
            for (int t : prompt) std::cout << data.idx2char[t];
            ctx = prompt;
            for (int i = 0; i < con.generated_text_size; i++)
            {
                int next = trans.predict_next(ctx, con.temperature);
                std::cout << data.idx2char[next];
                std::cout.flush();
                ctx.erase(ctx.begin());
                ctx.push_back(next);
            }
            std::cout << "\"\n\n==========";

            if (con.synthetic)
            {
                std::cout << "\"\n\n========== Generated Synthetic: \n\n\"";
                std::mt19937 rng(999);

                for (int i = 0; i < 10; i++)
                {
                    std::uniform_int_distribution<int> dist(0, (int)data.lines.size() - 1);
                    const ivec& row = data.lines[dist(rng)];

                    ivec ctx(row.begin(), row.end() - 1);
                    for (int t : ctx) std::cout << data.idx2char[t];

                    int next = trans.predict_next_best(ctx);
                    //std::cout << data.idx2char[next];
                    std::cout << " p-t = " << data.idx2char[next];

                    std::cout << " - " << data.idx2char[row[row.size() - 1]];

                    std::cout << " = " << abs((next - row[row.size() - 1]));

                    std::cout << "\n";
                }
                std::cout << "\"\n\n==========";
            }

            // validation loss
            std::mt19937 rng(999);
            atype test_loss = 0.0;
            for (int i = 0; i < 10; i++)
            {
                ivec inp, tgt; std::tie(inp, tgt) = data.getBatch(rng);
                test_loss += cross_entropy_loss(trans.forward(inp), tgt);
            }
            std::cout << "\n\n========== Test avg validation loss without dropout (10 seq): " << test_loss / 10 << "\n\n";

            con.train = true;
        }
        void run(std::string txt)
        {
            int nthreads = (con.num_threads > 0) ? con.num_threads : omp_get_max_threads();
            std::cout << "Transformer Praim (C++) - ver 1.0\n"
                << "\n dim           = " << con.dim
                << "\n layers        = " << con.layers
                << "\n heads         = " << con.heads
                << "\n seq_length    = " << con.seq_length
                << "\n batch         = " << con.batch_size
                << "\n"

                << "\n dropout_AHead = " << con.dropout_AttHead
                << "\n dropout_MHA   = " << con.dropout_AttMHA
                << "\n dropout_FFN   = " << con.dropout_FFN
                << "\n"

                << "\n lr            = " << con.lr
                << "\n"

                << "\n weight_decay  = " << con.weight_decay
                << "\n AdamW b1      = " << con.adamw_beta1
                << "\n AdamW b2      = " << con.adamw_beta2

                << "\n"
                << "\n epochs        = " << con.epochs
                << "\n threads       = " << nthreads << "\n\n";

            data.loadFromFile(txt);
            data.prepare();
            trans.init();
            train();
            test();
        }
    } pre;

}

int main()
{
    // in project settings: set = C / C++ → Language → OpenMP Support → Yes
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    std::string txt = "shakespeare_char.txt";
    nm51::pre.run(txt);

    system("pause");
    return 0;
}
