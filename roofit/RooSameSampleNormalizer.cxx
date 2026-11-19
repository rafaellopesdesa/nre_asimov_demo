#include "RooSameSampleNormalizer.h"
#include <RooArgList.h>
#include <RooFit/Detail/MathFuncs.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

ClassImp(RooSameSampleNormalizer)

RooSameSampleNormalizer::RooSameSampleNormalizer(
    const char *name, const char *title, const RooArgList &parameters,
    const std::vector<double> &measure, const std::vector<double> &low,
    const std::vector<double> &high)
    : RooAbsReal(name, title), measure_(measure), low_(low), high_(high) {
  parameters_.add(parameters);
  if (measure_.empty() || low_.size() != measure_.size()*parameters_.size() || high_.size()!=low_.size())
    throw std::invalid_argument("Same-sample normalizer array dimensions");
  for (double x : measure_) if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite reference measure");
  for (const auto *v : {&low_, &high_})
    for (double x : *v) if (!(x>0) || !std::isfinite(x)) throw std::invalid_argument("Nonpositive interpolation anchor");
}

RooSameSampleNormalizer::RooSameSampleNormalizer(const RooSameSampleNormalizer &other, const char *name)
    : RooAbsReal(other, name), parameters_("parameters", this, other.parameters_),
      measure_(other.measure_), low_(other.low_), high_(other.high_) {}

void RooSameSampleNormalizer::initializeCache() const {
  polynomial_.resize(low_.size());
  for (std::size_t k=0; k<low_.size(); ++k) {
    const double l=low_[k], h=high_[k], lh=std::log(h), ll=std::log(l);
    const double S0=(h+l)/2, A0=(h-l)/2;
    const double S1=(h*lh-l*ll)/2, A1=(h*lh+l*ll)/2;
    const double S2=(h*lh*lh+l*ll*ll)/2, A2=(h*lh*lh-l*ll*ll)/2;
    // ROOT 6.34 MathFuncs::flexibleInterpSingle(5,...), nominal=boundary=1.
    polynomial_[k] = {(15*A0-7*S1+A2)/8, (-24+24*S0-9*A1+S2)/8,
      (-5*A0+5*S1-A2)/4, (12-12*S0+7*A1-S2)/4,
      (3*A0-3*S1+A2)/8, (-8+8*S0-5*A1+S2)/8, ll, lh};
  }
  previous_.assign(parameters_.size(), std::numeric_limits<double>::quiet_NaN());
  factors_.assign(low_.size(), 1.);
  product_.assign(measure_.size(), 1.);
  updates_=0;
}

double RooSameSampleNormalizer::evaluate() const {
  if (polynomial_.empty()) initializeCache();
  const std::size_t n=measure_.size();
  bool rebase=(++updates_%128==0);
  for (std::size_t j=0; j<parameters_.size(); ++j) {
    const double x=static_cast<const RooAbsReal*>(parameters_.at(j))->getVal();
    if (x==previous_[j]) continue;
    previous_[j]=x;
    for (std::size_t i=0; i<n; ++i) {
      const std::size_t k=j*n+i;
      const auto &a=polynomial_[k];
      const double next=x>=1 ? std::exp(x*a[7]) : x<=-1 ? std::exp(-x*a[6]) :
        1+x*(a[0]+x*(a[1]+x*(a[2]+x*(a[3]+x*(a[4]+x*a[5])))));
      if (factors_[k]==0 || !std::isfinite(factors_[k])) rebase=true;
      else product_[i]*=next/factors_[k];
      factors_[k]=next;
    }
  }
  // Rebase periodically to bound roundoff from incremental updates. The cache
  // is only an optimization; it never freezes a parameter-dependent factor.
  if (rebase) {
    std::fill(product_.begin(), product_.end(), 1.);
    for (std::size_t j=0; j<parameters_.size(); ++j)
      for (std::size_t i=0; i<n; ++i) product_[i]*=factors_[j*n+i];
  }
  return std::inner_product(measure_.begin(), measure_.end(), product_.begin(), 0.);
}

double RooSameSampleNormalizer::directValue() const {
  const std::size_t n=measure_.size();
  double sum=0;
  for (std::size_t i=0; i<n; ++i) {
    double value=1;
    for (std::size_t j=0; j<parameters_.size(); ++j)
      value+=RooFit::Detail::MathFuncs::flexibleInterpSingle(5, low_[j*n+i], high_[j*n+i], 1., 1.,
          static_cast<const RooAbsReal*>(parameters_.at(j))->getVal(), value);
    sum+=measure_[i]*value;
  }
  return sum;
}
