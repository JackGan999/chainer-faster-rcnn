# A Faster R-CNN Implementation using Chainer

An experimental repository in progress with Chainer code to run the Faster R-CNN. The method was originally proposed by Shaoqing Ren et al. in June, 2015 and is one of the best performing object detection and classification algorithms in terms of speed performance and accuracy, at the time of writing this in April 2016.

## References

- [Original paper, arXiv](http://arxiv.org/abs/1506.01497)
- [Original MATLAB implementaion, GitHub](https://github.com/ShaoqingRen/faster_rcnn)
- [Python implementation (~10% slower), GitHub](https://github.com/rbgirshick/py-faster-rcnn)

## Challenges

- Implement the forward and backward methods for the Multi-Task Loss using Chainer.
  - First, make it work. Then optimize for performance in terms of speed.
  - Consider implementing the CPU version first if it seems easier.
- Data caching, reuse on GPU for performance reasons.
  - E.g. initial anchor creation.

