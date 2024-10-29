#include <algorithm>
#include <functional>
#include <assert.h>
#include <cmath>
#include <vector>


/**
 * 统计工具类
 */
template<typename TELEM>
class SimpleStats
{
public:
    SimpleStats(): _sum(0), _sqr_sum(0) {}
    ~SimpleStats() {}

    /**
     * 添加单个数据点
     * @param val 待添加的数据
     */
    inline void append(const TELEM& val) {
        if (_values.empty()) {
            _max = _min = val;
        } else {
            _max = std::max(val, _max);
            _min = std::min(val, _min);
        }
        _sum += val;
        _sqr_sum += val*val;
        _values.emplace_back(val);
    }

    /**
     * 添加多个数据
     * @param p_elems 待添加数据数组的首地址指针
     * @param n 待添加的数据个数
     */
    inline void append_many(TELEM* p_elems, const size_t& n) {
        if (!p_elems || n <= 0) {
            return;
        }
        for (auto i=0; i<n; i++) {
            append(p_elems[i]);
        }
    }

    /**
     * 添加多个数据，支持从使用转换函数从原始数据提取
     * @param p_elems 原始数据数组首地址指针
     * @param n 待添加的数据个数
     * @param converter 数据转换回调函数，入参两个：第一个是数据索引，第二个是原始数据，返回提取的被统计数据
     */
    template<typename TSRC>
    inline void append_many(TSRC* p_elems, const size_t& n, std::function<TELEM(size_t, const TSRC&)>&& converter) {
        if (!p_elems || n <= 0) {
            return;
        }
        for (size_t i=0; i<n; i++) {
            TELEM e = converter(i, p_elems[i]);
            append(e);
        }
    }

    /**
     * 返回当前数据个数
     */
    inline size_t count() const {
        return _values.size();
    }

    /**
     * 返回当前数据最小值
     */
    inline TELEM min() const {
        assert(count() > 0);
        return _min;
    }

    /**
     * 返回当前数据最大值
     */
    inline TELEM max() const {
        assert(count() > 0);
        return _max;
    }

    /**
     * 返回当前数据平均值
     */
    inline TELEM mean() const {
        assert(count() > 0);
        return _sum / count();
    }

    /**
     * 返回当前数据方差
     */
    inline TELEM var() const {
        assert(count() > 0);
        return _sqr_sum/count() - mean()*mean();
    }

    /**
     * 返回当前数据标准差
     */
    inline TELEM std() const {
        return std::sqrt(var());
    }

    inline const std::vector<TELEM>& data() const {
        return _values;
    }

protected:
    std::vector<TELEM> _values;
    TELEM _max;
    TELEM _min;
    TELEM _sum;
    TELEM _sqr_sum;
};
