#include <algorithm>
#include <cfloat>
#include <vector>

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (pool_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = pool_param.kernel_size();
  } else {
    kernel_h_ = pool_param.kernel_h();
    kernel_w_ = pool_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
  

}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  (*top)[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top->size() > 1) {
    (*top)[1]->ReshapeLike(*(*top)[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top->size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
  //If structural pooling we wil initialize the pooling_structure
  if (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_STRUCT_SEL) {
    //LoadPoolingStructure();
    //CHECK_EQ(this->blobs_.size(), 0) << "PoolingLayer should have no blobs unless STRUCT_SEL.";

    //this->blobs_.resize(1);
    
    //CHECK_EQ(this->blobs_.size(), 1) << "PoolingLayer should now have a blob to hold the STRUCT_SEL pooling structure.";
    
    // Intialize the pooling_structure (we do the grid-based approach so the size is the following)
    //this->blobs_[0].reset(new Blob<Dtype>(channels_, pooled_height_*pooled_width_/*map_size*/, pooled_height_, pooled_width_));
    //this->blobs_[0]->Reshape(channels_, pooled_height_*pooled_width_/*map_size*/, pooled_height_, pooled_width_);
    
    //pooling_structure_ = this->blobs_[0].get();
    pooling_structure_.Reshape(channels_, pooled_height_*pooled_width_/*map_size*/, pooled_height_, pooled_width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int* pooling_structure;
  
  const int top_count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += (*top)[0]->offset(0, 1);
        } else {
          mask += (*top)[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STRUCT_SEL:
    GeneratePoolingStructure();
    pooling_structure  = pooling_structure_.mutable_cpu_data();

    // Initialize
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = -FLT_MAX;
    }

    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
       pooling_structure  = pooling_structure_.mutable_cpu_data();
         for (int c = 0; c < channels_; ++c) {
            //for (int nk = 0; nk < pooled_height_*pooled_width_; ++nk) //go through the neuron maps
	    //{
	   for (int ph = 0; ph < pooled_height_; ++ph) {
	      for (int pw = 0; pw < pooled_width_; ++pw) {
		int hstart = ph * stride_h_ - pad_h_;
		int wstart = pw * stride_w_ - pad_w_;
		int hend = min(hstart + kernel_h_, height_ + pad_h_);
		int wend = min(wstart + kernel_w_, width_ + pad_w_);
//		int pool_size = (hend - hstart) * (wend - wstart);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height_);
		wend = min(wend, width_);
		if(Caffe::phase() == Caffe::TRAIN){
		  if(pooling_structure[ph * pooled_width_ + pw] == 1){
		    for (int h = hstart; h < hend; ++h) {
		      for (int w = wstart; w < wend; ++w) {
			  top_data[ph * pooled_width_ + pw]  = max( top_data[ph * pooled_width_ + pw], bottom_data[h * width_ + w]);
		      }
		    }
		  }else{
		    top_data[ph * pooled_width_ + pw]  = 0;
		  }
		  
		  pooling_structure += pooling_structure_.offset(0,1); //move through map neurons
		  
		}else{
		  for (int h = hstart; h < hend; ++h) {
		    for (int w = wstart; w < wend; ++w) {
			top_data[ph * pooled_width_ + pw]  = max( top_data[ph * pooled_width_ + pw], bottom_data[h * width_ + w]);
		    }
		  }
		}
		
		
	      }
	    }
	    bottom_data += bottom[0]->offset(0, 1);
	    top_data += (*top)[0]->offset(0, 1);
	}
    }

    //set those already not set to 0
    for (int i = 0; i < top_count; ++i) {
      if(top_data[i] == -FLT_MAX)
          top_data[i] = Dtype(0.);
    }
    break;

  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int* pooling_structure;
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STRUCT_SEL:
    
    for (int n = 0; n < top[0]->num(); ++n) {
      pooling_structure  = pooling_structure_.cpu_data();
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            //int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            
	    if(Caffe::phase() == Caffe::TRAIN){
	      if(pooling_structure[ph * pooled_width_ + pw] == 1){
		for (int h = hstart; h < hend; ++h) {
		  for (int w = wstart; w < wend; ++w) {
		    bottom_diff[h * width_ + w] +=
		      top_diff[ph * pooled_width_ + pw];
		  }
		}
	      }
	      
	      pooling_structure += pooling_structure_.offset(0,1); //move through map neurons
	    }else{
	      for (int h = hstart; h < hend; ++h) {
		for (int w = wstart; w < wend; ++w) {
		  bottom_diff[h * width_ + w] +=
		    top_diff[ph * pooled_width_ + pw];
		}
	      }
	    
	    }
          }
        }
        // offset
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    
    break;
    
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

// Defined in [upper, lower] 
int alea(int lower, int upper){

  // Using Boost libraries:
  boost::random::mt19937 gen(static_cast<unsigned int>(std::time(0)));
  boost::random::uniform_int_distribution<> rngBoost(lower, upper);
  
  return rngBoost(gen);
  
  // Using c++11 random libraries:
  /*srand (time(NULL));
  upper++;
  int range= upper - lower;
  return rand() % range + lower;*/
  //return (upper-lower)/2;

}


template<typename Dtype> 
void PoolingLayer<Dtype>::GenerateSinglePoolingMask(int* pooling_structure, float alpha, bool switch_off_rect) {
  
  //int W, int H, unsigned int **mask;

  //int n_pooled_elements, c, i, k;
  int w, h;

  int top_size = pooled_width_*pooled_height_;//36; Area
  
  int W = pooled_width_;
  int H = pooled_height_;
  
  
  //CHECK_EQ(this->blobs_.size(), 0) << "PoolingLayer should have no blobs unless STRUCT_SEL.";

  //this->blobs_.resize(1);
  
  // Intialize the pooling_structure
  //this->blobs_[0].reset(new Blob<Dtype>(
//	    channels_, pooled_height_*pooled_width_/*map_size*/, height_, width_));

  
  // we do the grid-based approach where we mask top-level neurons
  int Area = top_size;
  
  int n_ave_zero_elements= (int) ((float) Area * alpha + 0.4999999);
  int min_w= 2;
  int max_w= pooled_width_ - 1;
  int half_range= (int) ((float) n_ave_zero_elements * 0.25 + 0.4999999);
  
  int min_a= (n_ave_zero_elements - half_range > min_w * min_w) ? n_ave_zero_elements - half_range : min_w * min_w;
  int max_a= (n_ave_zero_elements + half_range < max_w * max_w) ? n_ave_zero_elements + half_range : max_w * max_w;
  int half_a= (n_ave_zero_elements - min_a < max_a - n_ave_zero_elements) ? n_ave_zero_elements - min_a : max_a - n_ave_zero_elements;

  int m_a= n_ave_zero_elements - half_a;
  int M_a= n_ave_zero_elements + half_a;

  int A_r= alea(m_a, M_a);
  float q= sqrt( (float) A_r);
  int qi= (int) q + 1;
  
  float min_e= FLT_MAX;
  int min_d= (int) (q + 0.4999999);
  for (int d= 2; d < qi; d++) {
    int d1= (int) ((float) A_r / d + 0.4999999);
      if (d1 <= max_w) {
	float e= A_r - d1 * d;
	e= (e >= 0) ? e : -e;

	if (min_e >= e) {
	  min_e= e;
	  min_d= d;
      }
    }
  }
    
  if (alea(0, 1) == 1) {    
    w= min_d;
    if(min_d != 0)
	h= (int) ((float) A_r / min_d + 0.4999999);
  }
  else {
    h= min_d;
    if(min_d != 0)
	w= (int) ((float) A_r / min_d + 0.4999999);  
  }


  int x= alea(0, W - w);
  int y= alea(0, H - h);
  int x1= x + w;
  int y1= y + h;
    
  unsigned int inner_bit= 1;
  unsigned int outer_bit= 0;
  if (switch_off_rect) {
	inner_bit= 0;
	outer_bit= 1;
  }
  
  //int x_coordinate, y_coordinate;
  	
  for (int i= 0; i < pooled_height_; i++){
	for (int j= 0; j < pooled_width_; j++){
	  
	  //take care not to overstep the boundaries
	  //x_coordinate = min(j, pooled_width_- 1);
	  //y_coordinate = min(i, pooled_height_- 1);
		
	  if (i >= y && i < y1 && j >= x && j < x1)
	    pooling_structure[i*pooled_width_ + j] = inner_bit;
	  else
	    pooling_structure[i*pooled_width_ + j] = outer_bit;
	}
  }

}

template<typename Dtype> 
void PoolingLayer<Dtype>::GeneratePoolingStructure(float alpha, bool switch_off_rect) {
  
  int n_pooled_elements, c, i;

  int top_size = pooled_width_*pooled_height_;//36;
  
  //CHECK_EQ(this->blobs_.size(), 0) << "PoolingLayer should have no blobs unless STRUCT_SEL.";

  //this->blobs_.resize(1);
  
  // Intialize the pooling_structure (we do the grid-based approach so the size is the following)
  //this->blobs_[0].reset(new Blob<Dtype>(
//	    channels_, pooled_height_*pooled_width_/*map_size*/, pooled_height_, pooled_width_));

  //pooling_structure_ = this->blobs_[0].get();

  //fill the structure with zeros
  //FillerParameter filler_param;
  //filler_param.set_value(0.);
  //ConstantFiller<Dtype> filler(filler_param);
  //filler.Fill(&pooling_structure_);
  // shoud it be with the num? pooling_structure_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);-]
  ///pooling_structure_.Reshape(channels_, pooled_height_*pooled_width_/*map_size*/, height_, width_); // I will use masks instead of indexing for efficiency-]
  //pooling_structure_.Update();
  
  CHECK_EQ(this->layer_param_.pooling_param().pool(), PoolingParameter_PoolMethod_STRUCT_SEL) << "PoolingLayer should not generate pooling mask unless STRUCT_SEL.";
  
  int* pooling_structure = pooling_structure_.mutable_cpu_data();

  //int x_coordinate, y_coordinate;
  
  //load the mutable structure with ones where necessary
  for (c = 0; c < channels_; ++c) {
    for (i = 0; i < top_size; ++i) {
      
      //fill it with zeros-]
      /*for (int ph = 0; ph < height_; ++ph) {-]
	      for (int pw = 0; pw < width_; ++pw) {-]
		      pooling_structure[ph * width_ + pw] = 0;-]
	      }-]
      }*/
      
      //inFile >> n_pooled_elements;

      /*for (k = 0; k < n_pooled_elements; ++k) {
	inFile >> x_coordinate;
	inFile >> y_coordinate;
	
	//take care not to overstep the boundaries
	x_coordinate = min(x_coordinate, width_- 1);
	y_coordinate = min(y_coordinate, height_- 1);
	
	pooling_structure[y_coordinate*width_ + x_coordinate] = 1;
      
      }*/

      GenerateSinglePoolingMask(pooling_structure, alpha, false);
      
      pooling_structure += pooling_structure_.offset(0,1); //move through map neurons-]
    }
  }
   
  
}
 
#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);


}  // namespace caffe
