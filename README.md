# CUDA C++ Programming Guide 한국어

<br>

## [목차](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html#contents)
- [1.1 GPU  사용의  이점](#11-gpu-사용의-이점) [[원본]](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#the-benefits-of-using-gpus)
- [1.2 범용 병렬 컴퓨팅 플랫폼 및 프로그래밍 모델]() [[원본]](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-a-general-purpose-parallel-computing-platform-and-programming-model)
- [1.3 확장 가능한 프로그래밍 모델]() [[원본]](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#a-scalable-programming-model)
- [1.4 문서 구조]() [[원본]](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#document-structure)


## [1. 소개](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)

## 1.1 GPU 사용의 이점


GPU(그래픽 처리 장치)는 비슷한 가격과 전력 범위 내에서 CPU보다 훨씬 높은 명령 처리량과 메모리 대역폭을 제공합니다. 많은 애플리케이션이 이러한 높은 기능을 활용하여 CPU보다 GPU에서 더 빠르게 실행됩니다([GPU 애플리케이션 참조](https://www.nvidia.com/en-us/gpu-accelerated-applications/)). FPGA와 같은 다른 컴퓨터 장치도 에너지 효율이 높지만 GPU 보다 낮은 프로그래밍 유연성을 제공합니다.
<br>
<br>
GPU와 CPU의 이러한 기능 차이는 서로 다른 목표를 염두에 두고 설계되었기 때문에 존재합니다. CPU는 스레드라고 하는 일련의 작업을 가능한 한 빨리 실행하는 데 탁월하도록 설계되었으며 이러한 스레드를 몇 개 병렬로 실행할 수 있는 반면 GPU는 수천 개의 스레드를 병렬로 실행하는 데 탁월하도록 설계되었습니다(더 느린 단일 스레드 성능을 조정하여 처리량을 향상).
<br>
<br>
GPU는 고도의 병렬 연산에 특화되어 있으므로 데이터 캐싱 및 흐름 제어보다는 데이터 처리에 더 많은 트랜지스터가 사용되도록 설계되었습니다. 그림 1은 CPU와 GPU에 대한 칩 리소스의 분포 예시를 보여줍니다.
<p align="center">

  <img src="https://github.com/JeHeeYu/One-Hundred-ME/assets/87363461/6a9c135c-be14-4e2d-91e0-9caf61d55e7a">
  <br>
  그림 1: GPU는 데이터 처리에 더 많은 트랜지스터를 사용
  
</p>
<br>
부동 소수점 계산과 같은 데이터 처리에 더 많은 트랜지스터를 사용하는 것은 고도의 병렬 계산에 도움이 됩니다. GPU는 트랜지스터 측면에서 비용이 많이 드는 긴 메모리 액세스 대기 시간을 피하기 위해 대용량 데이터 캐시와 복잡한 흐름 제어에 의존하는 대신 계산으로 메모리 액세스 대기 시간을 줄일 수 있습니다.
<br>
<br>
일반적으로 애플리케이션은 병렬 부분과 순차 부분이 혼합되어 있으므로 전체 성능을 극대화하기 위해 시스템은 GPU와 CPU를 혼합하여 설계됩니다. 높은 수준의 병렬 처리 기능을 갖춘 애플리케이션은 GPU의 대규모 병렬 특성을 활용하여 CPU보다 더 높은 성능을 달성할 수 있습니다.

<br>
<br>

## 1.2 범용 병렬 컴퓨팅 플랫폼 및 프로그래밍 모델
2006년 11월, NVIDIA®는 NVIDIA GPU의 병렬 컴퓨팅 엔진을 활용하여 여러 가지 복잡한 컴퓨팅 문제를 CPU보다 효율적인 방식으로 해결하는 범용 병렬 컴퓨팅 플랫폼이자 프로그래밍 모델인 CUDA®를 출시했습니다.
<br>
<br>
CUDA는 개발자들이 C++를 고급 프로그래밍 언어로 사용할 수 있는 소프트웨어 환경을 제공합니다. 그림 2와 같이 FORTRAN, DirectCompute, OpenACC와 같은 다른 언어, 플리케이션 프로그래밍 인터페이스 또는 디렉티브 기반 접근 방식이 지원됩니다.
<p align="center">

  <img src="https://github.com/JeHeeYu/CUDA-Cpp-Programming-Guide-Korean/assets/87363461/4a282abe-9854-47f9-8243-3acda21a50b1">
  <br>
  그림 2: GPU 컴퓨팅 애플리케이션 CUDA는 다양한 언어와 애플리케이션 프로그래밍 인터페이스를 지원하도록 설계되었습니다.
  
</p>
<br>
<br>

## 1.3 확장 가능한 프로그래밍 모델
멀티코어 CPU와 매니코어 GPU의 등장은 주류 프로세서 칩이 이제 병렬 시스템이 되었음을 의미합니다. 문제는 3D 그래픽 애플리케이션이 매우 다양한 코어 수를 사용하여 병렬성을 매니코어 GPU로 투명하게 확장하는 것과 마찬가지로 증가하는 프로세서 코어 수를 활용하기 위해 병렬성을 투명하게 확장하는 애플리케이션 소프트웨어를 개발하는 것입니다.
