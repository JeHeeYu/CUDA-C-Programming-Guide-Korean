# CUDA C++ Programming Guide 한국어

<br>

## [목차](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html#contents)
- ### [1. 소개](#1-소개)
  - [1.1 GPU  사용의  이점](#11-gpu-사용의-이점) [[원본]](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#the-benefits-of-using-gpus)
  - [1.2 범용 병렬 컴퓨팅 플랫폼 및 프로그래밍 모델](#12-범용-병렬-컴퓨팅-플랫폼-및-프로그래밍-모델) [[원본]](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-a-general-purpose-parallel-computing-platform-and-programming-model)
  - [1.3 확장 가능한 프로그래밍 모델](#13-확장-가능한-프로그래밍-모델) [[원본]](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#a-scalable-programming-model)
  - [1.4 문서 구조](#14-문서-구조) [[원본]](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#document-structure)

- ### [2. 프로그래밍 모델](#2-프로그래밍-모델-1)
  - [2.1. 커널](#21-커널)
  - [2.2. 스레드 계층구조](#22-스레드-계층구조)
    - [2.2.1. 스레드 블록 클러스터]()
  - [2.3 메모리 계층구조](#그림-6-메모리-계층구조)
- ### [3. 프로그래밍 인터페이스](#3-프로그래밍-인터페이스-1)
  - 3.2.
    - [3.2.4 공유 메모리](#324-공유-메모리)
    - [3.2.5 분산 공유 메모리](#325-분산-공유-메모리)
- ### [7. C++ 언어 확장](#7-c-언어-확장-1)
- ### [8. 협력 그룹](#8-협력-그룹-1)

## 1. 소개

## 1.1 GPU 사용의 이점

GPU(그래픽 처리 장치)는 비슷한 가격과 전력 범위 내에서 CPU보다 훨씬 높은 명령 처리량과 메모리 대역폭을 제공합니다. 많은 애플리케이션이 이러한 높은 기능을 활용하여 CPU보다 GPU에서 더 빠르게 실행됩니다([GPU 애플리케이션 참조](https://www.nvidia.com/en-us/gpu-accelerated-applications/)). FPGA와 같은 다른 컴퓨터 장치도 에너지 효율이 높지만 GPU 보다 낮은 프로그래밍 유연성을 제공합니다.
<br>
<br>
GPU와 CPU의 이러한 기능 차이는 서로 다른 목표를 염두에 두고 설계되었기 때문에 존재합니다. CPU는 스레드라고 하는 일련의 작업을 가능한 한 빨리 실행하는 데 탁월하도록 설계되었으며 이러한 스레드를 몇 개 병렬로 실행할 수 있는 반면 GPU는 수천 개의 스레드를 병렬로 실행하는 데 탁월하도록 설계되었습니다(더 느린 단일 스레드 성능을 조정하여 처리량을 향상).
<br>
<br>
GPU는 고도의 병렬 연산에 특화되어 있으므로 데이터 캐싱 및 흐름 제어보다는 데이터 처리에 더 많은 트랜지스터가 사용되도록 설계되었습니다. [그림 1](#그림-1-gpu는-데이터-처리에-더-많은-트랜지스터를-사용)은 CPU와 GPU에 대한 칩 리소스의 분포 예시를 보여줍니다.
<p align="center">

  <img src="https://github.com/JeHeeYu/One-Hundred-ME/assets/87363461/6a9c135c-be14-4e2d-91e0-9caf61d55e7a">
  <br>
  <h5 align="center">그림 1: GPU는 데이터 처리에 더 많은 트랜지스터를 사용</h5>

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
CUDA는 개발자들이 C++를 고급 프로그래밍 언어로 사용할 수 있는 소프트웨어 환경을 제공합니다. [그림 2](#그림-2-gpu-컴퓨팅-애플리케이션-cuda는-다양한-언어와-애플리케이션-프로그래밍-인터페이스를-지원하도록-설계되었습니다)와 같이 FORTRAN, DirectCompute, OpenACC와 같은 다른 언어, 플리케이션 프로그래밍 인터페이스 또는 디렉티브 기반 접근 방식이 지원됩니다.
<p align="center">

  <img src="https://github.com/JeHeeYu/CUDA-Cpp-Programming-Guide-Korean/assets/87363461/4a282abe-9854-47f9-8243-3acda21a50b1">
  <br>
  <h5 align="center">그림 2: GPU 컴퓨팅 애플리케이션 CUDA는 다양한 언어와 애플리케이션 프로그래밍 인터페이스를 지원하도록 설계되었습니다.</h5>

</p>
<br>
<br>

## 1.3 확장 가능한 프로그래밍 모델
멀티코어 CPU와 매니코어 GPU의 등장은 주류 프로세서 칩이 이제 병렬 시스템이 되었음을 의미합니다. 문제는 3D 그래픽 애플리케이션이 매우 다양한 코어 수를 사용하여 병렬성을 매니코어 GPU로 투명하게 확장하는 것과 마찬가지로 증가하는 프로세서 코어 수를 활용하기 위해 병렬성을 투명하게 확장하는 애플리케이션 소프트웨어를 개발하는 것입니다.
<br>
<br>
CUDA 병렬 프로그래밍 모델은 C와 같은 표준 프로그래밍 언어에 익숙한 프로그래머를 위해 낮은 학습 곡선을 유지하면서 이러한 문제를 극복하도록 설계되었습니다.
<br>
<br>
그 핵심에는 스레드 그룹의 계층 구조, 공유 메모리 및 장벽 동기화와 같은 세 가지 주요 추상화가 있으며, 이는 최소한의 언어 확장 세트로 프로그래머에게 간단히 노출됩니다.
<br>
<br>
이러한 추상화는 세분화된 데이터 병렬 및 스레드 병렬을 제공하며, 세분화된 데이터 병렬 처리 및 스레드 병렬 처리를 제공합니다. 이 가이드는 프로그래머가 문제를 스레드 블록에 의해 독립적으로 병렬로 해결될 수 있는 대략적인 하위 문제로 분할하고 각 하위 문제를 블록 내의 모든 스레드에 의해 병렬로 해결될 수 있는 더 미세한 조각으로 분할하도록 안내합니다.
<br>
<br>
이러한 분해는 각 하위 문제를 해결할 때 스레드가 협력할 수 있도록 함으로써 언어 표현력을 보존하는 동시에 자동 확장성을 가능하게 합니다. 실제로, 각각의 스레드 블록은 GPU 내의 사용 가능한 멀티프로세서에 임의의 순서로, 동시에 또는 순차적으로 스케줄링될 수 있으므로 컴파일된 CUDA 프로그램은 [그림 3](#그림-3-자동-확장성)과 같이 임의의 수의 멀티프로세서에서 실행될 수 있으며, 런타임 시스템만이 물리적 멀티프로세서 수를 알면 됩니다.
<br>
<br>
이 확장 가능한 프로그래밍 모델을 통해 GPU 아키텍처는 멀티프로세서 및 메모리 파티션 수를 확장함으로써 광범위한 시장 범위에 적용할 수 있습니다: 고성능 매니아 GeForce GPU와 전문가용 Quadro 및 Tesla 컴퓨팅 제품부터 다양한 저가의 주류 GeForce GPU에 이르기까지(모든 [CUDA 지원 GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus) 목록은 CUDA 지원 GPU 참조)
<p align="center">

  <img src="https://github.com/JeHeeYu/One-Hundred-ME/assets/87363461/6a9c135c-be14-4e2d-91e0-9caf61d55e7a">
  <br>
  <h5 align="center">그림 1: GPU는 데이터 처리에 더 많은 트랜지스터를 사용</h5>

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
CUDA는 개발자들이 C++를 고급 프로그래밍 언어로 사용할 수 있는 소프트웨어 환경을 제공합니다. [그림 2](#그림-2-gpu-컴퓨팅-애플리케이션-cuda는-다양한-언어와-애플리케이션-프로그래밍-인터페이스를-지원하도록-설계되었습니다)와 같이 FORTRAN, DirectCompute, OpenACC와 같은 다른 언어, 플리케이션 프로그래밍 인터페이스 또는 디렉티브 기반 접근 방식이 지원됩니다.
<p align="center">

  <img src="https://github.com/JeHeeYu/CUDA-Cpp-Programming-Guide-Korean/assets/87363461/4a282abe-9854-47f9-8243-3acda21a50b1">
  <br>
  <h5 align="center">그림 2: GPU 컴퓨팅 애플리케이션 CUDA는 다양한 언어와 애플리케이션 프로그래밍 인터페이스를 지원하도록 설계되었습니다.</h5>

</p>
<br>
<br>

## 1.3 확장 가능한 프로그래밍 모델
멀티코어 CPU와 매니코어 GPU의 등장은 주류 프로세서 칩이 이제 병렬 시스템이 되었음을 의미합니다. 문제는 3D 그래픽 애플리케이션이 매우 다양한 코어 수를 사용하여 병렬성을 매니코어 GPU로 투명하게 확장하는 것과 마찬가지로 증가하는 프로세서 코어 수를 활용하기 위해 병렬성을 투명하게 확장하는 애플리케이션 소프트웨어를 개발하는 것입니다.
<br>
<br>
CUDA 병렬 프로그래밍 모델은 C와 같은 표준 프로그래밍 언어에 익숙한 프로그래머를 위해 낮은 학습 곡선을 유지하면서 이러한 문제를 극복하도록 설계되었습니다.
<br>
<br>
그 핵심에는 스레드 그룹의 계층 구조, 공유 메모리 및 장벽 동기화와 같은 세 가지 주요 추상화가 있으며, 이는 최소한의 언어 확장 세트로 프로그래머에게 간단히 노출됩니다.
<br>
<br>
이러한 추상화는 세분화된 데이터 병렬 및 스레드 병렬을 제공하며, 세분화된 데이터 병렬 처리 및 스레드 병렬 처리를 제공합니다. 이 가이드는 프로그래머가 문제를 스레드 블록에 의해 독립적으로 병렬로 해결될 수 있는 대략적인 하위 문제로 분할하고 각 하위 문제를 블록 내의 모든 스레드에 의해 병렬로 해결될 수 있는 더 미세한 조각으로 분할하도록 안내합니다.
<br>
<br>
이러한 분해는 각 하위 문제를 해결할 때 스레드가 협력할 수 있도록 함으로써 언어 표현력을 보존하는 동시에 자동 확장성을 가능하게 합니다. 실제로, 각각의 스레드 블록은 GPU 내의 사용 가능한 멀티프로세서에 임의의 순서로, 동시에 또는 순차적으로 스케줄링될 수 있으므로 컴파일된 CUDA 프로그램은 [그림 3](#그림-3-자동-확장성)과 같이 임의의 수의 멀티프로세서에서 실행될 수 있으며, 런타임 시스템만이 물리적 멀티프로세서 수를 알면 됩니다.
<br>
<br>
이 확장 가능한 프로그래밍 모델을 통해 GPU 아키텍처는 멀티프로세서 및 메모리 파티션 수를 확장함으로써 광범위한 시장 범위에 적용할 수 있습니다: 고성능 매니아 GeForce GPU와 전문가용 Quadro 및 Tesla 컴퓨팅 제품부터 다양한 저가의 주류 GeForce GPU에 이르기까지(모든 [CUDA 지원 GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus) 목록은 CUDA 지원 GPU 참조)
<p align="center">

  <img src="https://github.com/JeHeeYu/CUDA-Cpp-Programming-Guide-Korean/assets/87363461/1eba8b21-ddac-4acf-9680-5405d13b5953">
  <br>
  <h5 align="center">그림 3: 자동 확장성</h5>

</p>
<br>
<br>

## 1.4 문서 구조
이 문서는 다음 섹션으로 구성되어 있습니다:

<br>
<br>

## 2. 프로그래밍 모델
이 장에서는 C++에서 CUDA 프로그래밍 모델이 어떻게 노출되는지 간략하게 설명하여 CUDA 프로그래밍 모델의 기본 개념을 소개합니다.
<br>
CUDA C++에 대한 자세한 설명은 [프로그래밍 인터페이스](#3-프로그래밍-인터페이스-1)에 나와 있습니다.
<br>
이 장과 다음에 사용된 벡터 추가 예제의 전체 코드는 [vectorAdd CUDA](https://docs.nvidia.com/cuda/cuda-samples/index.html#vector-addition) 샘플에서 찾을 수 있습니다.
<br>
<br>

## 2.1. 커널
CUDA C++는 프로그래머가 커널이라고 불리는 C++ 함수를 정의할 수 있게 함으로써 C++를 확장하는데, 이 함수는 호출될 때 일반 C++ 함수처럼 한 번만 실행되는 것이 아니라, N개의 다른 CUDA 스레드에 의해 N번 병렬로 실행됩니다.
<br>
<br>
커널은 __global__선언 지정자를 사용하여 정의되며 특정 커널 호출에 대해 해당 커널을 실행하는 CUDA 스레드 수는 새로운 <<...>>실행 구성 구문을 사용하여 지정됩니다([C++ 언어 확장](#7-c-언어-확장-1) 참조). 커널을 실행하는 각 스레드에는 내장 변수를 통해 커널 내에서 액세스할 수 있는 고유한 스레드 ID가 제공됩니다.
<br>
<br>

예를 들어, 다음 샘플 코드는 내장 변수 threadIdx를 사용하여 크기가 N인 두 벡터 A와 B를 추가하고 결과를 벡터 C에 저장합니다:
<br>

```
// 커널 정의
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
int main()
{
    ...
    // N개의 스레드를 사용한 커널 호출
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```
<br>
여기서 VecAdd()를 실행하는 N개의 스레드 각각은 한 쌍의 추가를 수행합니다.
<br>
<br>

## 2.2. 스레드 계층구조
편의상 threadIdx는 3성분 벡터이므로 1차원, 2차원, 3차원 스레드 인덱스를 사용하여 스레드를 식별할 수 있으며, 스레드 블록이라고 하는 1차원, 2차원 또는 3차원 스레드 블록을 형성합니다. 이것은 벡터, 행렬 또는 볼륨과 같은 도메인의 요소에 걸쳐 계산을 호출하는 자연스러운 방법을 제공합니다.
<br>
<br>
스레드의 인덱스와 스레드 ID는 간단한 방법으로 서로 연관됩니다: 1차원 블록의 경우 동일하며, 크기가 (Dx, Dy)인 2차원 블록의 경우 인덱스 (x, y)의 스레드 ID는 (x + y Dx)이고, 크기가 (Dx, Dy, Dz)인 3차원 블록의 경우 인덱스 (x, y, z)의 스레드 ID는 (x + y Dx + z Dx Dy)입니다.
<br>
<br>
예를 들어, 다음 코드는 NxN 크기의 두 행렬 A와 B를 추가하고 그 결과를 행렬 C에 저장합니다:
<br>

```
// 커널 정의
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // N * N * 1개의 스레드 블록으로 커널 호출
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```
<br>

블록의 모든 스레드는 동일한 스트리밍 멀티프로세서 코어에 존재할 것으로 예상되고 해당 코어의 제한된 메모리 리소스를 공유해야 하므로 블록당 스레드 수에 제한이 있습니다. 현재 GPU에서 스레드 블록은 최대 1024개의 스레드를 포함할 수 있습니다.
<br>
<br>
그러나 커널은 동일한 모양의 여러 스레드 블록에 의해 실행될 수 있으므로 총 스레드 수는 블록당 스레드 수에 블록 수를 곱한 값과 같습니다.
<br>
<br>
블록은 [그림 4](#그림-4-스레드-블록-그리드)에 표시된 것처럼 1차원, 2차원 또는 3차원 스레드 블록 그리드로 구성됩니다. 그리드의 스레드 블록 수는 일반적으로 처리되는 데이터의 크기에 따라 결정되며 일반적으로 시스템의 프로세서 수를 초과합니다.
<p align="center">
  <img src="https://github.com/JeHeeYu/CUDA-Cpp-Programming-Guide-Korean/assets/87363461/424107d0-d9fe-47c6-91a6-e337231cc0ae">
  <br>
  <h5 align="center">그림 4: 스레드 블록 그리드</h5>
  
</p>
<br>
<br>
<<...>> 구문에 지정된 블록당 스레드 수와 그리드당 블록 수는 type int 또는 dim3일 수 있습니다. 위의 예와 같이 2차원 블록 또는 그리드를 지정할 수 있습니다.
<br>
<br>
그리드 내의 각 블록은 내장된 blockIdx 변수를 통해 커널 내에서 접근 가능한 1차원, 2차원 또는 3차원 고유 인덱스로 식별할 수 있습니다. 스레드 블록의 크는 내장된 blockDim 변수를 통해 커널 내에서 액세스할 수 있습니다.
<br>
<br>
이전 MatAdd() 예제를 확장하여 여러 블록을 처리하면 코드는 다음과 같습니다.
<br>
  
```
// 커널 정의
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // 커널 호출
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```
<br>

이 경우에는 임의적이지만 16x16(256개 스레드)의 스레드 블록 크기가 일반적인 선택입니다. 그리드는 이전과 같이 행렬 요소당 하나의 스레드를 가질 만큼 충분한 블록으로 생성됩니다. 단순화를 위해 이 예제에서는 각 차원의 그리드당 스레드 수를 해당 차원의 블록당 스레드 수로 균등하게 나눌 수 있다고 가정하지만, 반드시 그럴 필요는 없습니다.
<br>
<br>
스레드 블록은 독립적으로 실행해야 합니다: 병렬 또는 직렬 등 어떤 순서로든 실행이 가능해야 합니다. 이러한 독립성 요구 사항을 통해 [그림 3](#그림-3-자동-확장성)에 표시된 것처럼 스레드 블록을 원하는 수의 코어에 걸쳐 원하는 순서로 예약할 수 있으므로 프로그래머는 코어 수에 따라 확장되는 코드를 작성할 수 있습니다.
<br>
<br>
블록 내의 스레드는 일부 공유 메모리를 통해 데이터를 공유하고 실행을 동기화하여 메모리 액세스를 조정함으로써 협력할 수 있습니다. 더 정확하게는 __synthreads() 고유 함수를 호출하여 커널에서 동기화 지점을 지정할 수 있습니다. __synthreads()는 블록의 모든 스레드가 계속 진행되도록 허용되기 전에 대기해야 하는 장벽 역할을 합니다. [공유 메모리](#324-공유-메모리)는 공유 메모리 사용의 예를 제공합니다. __syncthreads() 외에도 [Cooperative Groups API](#8-협력-그룹-1)는 다양한 스레드 동기화 기본 요소 제공합니다.
<br>
<br>
효율적인 협력을 위해 공유 메모리는 각 프로세서 코어 근처의 저지연 메모리(L1 캐시와 유사)가 될 것으로 예상되며 __syncthreads()는 경량일 것으로 예상됩니다.
<br>
<br>
## 2.2.1. 스레드 블록 클러스터
NVIDIA [Compute Capability 9.0](#168-compute-capability-90)이 도입되면서 CUDA 프로그래밍 모델에는 스레드 블록으로 구성된 스레드 블록 클러스터라는 선택적 계층 구조가 도입되었습니다. 스레드 블록의 스레드가 스트리밍 멀티프로세서에서 공동 스케줄링되는 방식과 유사하게 클러스터의 스레드 블록도 GPU의 GPC(GPU 처리 클러스터)에서 공동 스케줄링이 보장됩니다. 
<br>
<br>
클러스터도 스레드 블록과 마찬가지로 [그림 5](#그림-5-스레드-블록-클러스터의-그리드)와 같이 1차원, 2차원 또는 3차원으로 구성됩니다. 클러스터 내의 스레드 블록 수는 사용자가 정의할 수 있으며, CUDA에서 최대 8개의 스레드 블록을 휴대용 클러스터를 지원합니다. GPU 하드웨어 또는 MIG 구성이 8 멀티프로세서를 지원할 수 없을 만큼 작은 경우, 최대 클러스터 크기는 해당 하드웨어나 구성에 맞게 적절히 줄어듭니다. 이러한 작은 구성 및 8을 초과하는 스레드 블록 클러스터 크기를 지원하는 큰 구성의 식별은 아키텍처에 따라 다르며 cudaOccupancyMaxPotentialClusterSize API를 사용하여 조회할 수 있습니다.
<p align="center">
  <img src="https://github.com/JeHeeYu/CUDA-Cpp-Programming-Guide-Korean/assets/87363461/0af87e92-f609-4037-904d-93ec4b0645c3">
  <br>
  <h5 align="center">그림 5: 스레드 블록 클러스터의 그리드</h5>
  
</p>
<br>
<br>

> 클러스터 지원을 사용하여 시작된 커널에서 gridDim 변수는 호환성을 위해 여전히 스레드 블록 수를 기준으로 크기를 나타냅니다. [클러스터 그룹](#8412-클러스터-그룹) API를 사용하여 클러스터 내 블록의 순위를 확인할 수 있습니다.

<br>
스레드 블록 클러스터는 컴파일러 시간 커널 속성을 사용하거나 __cluster_dims_(X,Y,Z)를 사용 또는 CUDA 커널 시작 API cudaLaunchKernelEx를 사용하여 커널에서 사용할 수 있습니다. 아래 예제는 컴파일러 시간 커널 속성을 사용하여 클러스터를 시작하는 방법을 보여줍니다. 커널 속성을 사용하는 클러스터 크기는 컴파일 시 고정되며, 이후 <<<, >>>을 사용하여 커널을 시작할 수 있습니다. 커널이 컴파일 시간 클러스터 크기를 사용하는 경우 커널을 시작할 때 클러스터 크기를 수정할 수 없습니다.
<br>

```
// 커널 정의
// 컴파일 시간에 클러스터 크기는 X 차원에서 2이고 Y 및 Z 차원에서 1이 
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // 컴파일 시간에 설정된 클러스터 크기를 가진 커널 함수 호출
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // 그리드 차원은 클러스터 시작의 영향을 받지 않으며
    // 블록 수를 사용하여 열거됩니다.
    // 그리드 차원은 클러스터 크기의 배수여야 합니다.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```
<br>

스레드 블록 클러스터 크기는 런타임 시 설정할 수도 있으며 CUDA 커널 시작 API cudaLaunchKernelEx를 사용하여 커널을 시작할 수 있습니다. 아래 코드 예제는 확장 가능한 API를 사용하여 클러스터 커널을 시작하는 방법을 보여줍니다.
<br>

```
// 커널 정의의
// 커널에 컴파일 시간 속성이 부되지 않았습니다.
__global__ void cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // 런타임 클러스터 크기의 커널 호출
    {
        cudaLaunchConfig_t config = {0};
        // 그리드 차원은 클러스터 시작의 영향을 받지 않으며
        // 블록 수를 사용하여 열거됩니다.
        // 그리드 차원은 클러스터 크기의 배수여야 합니다.
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // X 차원의 클러스터 크기
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```

<br>
Compute capability 9.0이 있는 GPU에서는 클러스터의 모든 스레드 블록이 단일 GPU Processing Cluster(GPC)에서 공동 스케줄링되도록 보장하고 클러스터의 스레드 블록이 [클러스터 그룹](#8412-클러스터-그룹) API cluster.sync()를 사용하여 하드웨어 지원 동기화를 수행할 수 있도록 합니다. 클러스터 그룹은 각각 num_threads() 및 num_blocks() API를 사용하여 스레드 수 또는 블록 수를 기준으로 클러스터 그룹 크기를 쿼리하는 멤버 함수도 제공합니다. 클러스터 그룹의 스레드 또는 블록의 순위는 각각 dim_threads() 및 dim_blocks() API를 사용하여 조회할 수 있습니다.
<br>
<br>
클러스터에 속한 스레드 블록은 분산 공유 메모리에 액세스할 수 있습니다. 클러스터의 스레드 블록은 분산 공유 메모리 임의의 주소에 대한 읽기, 쓰기 및 atomics 을 수행할 수 있습니다. 분산 공유 메모리는 분산 공유 메모리에서 히스토그램을 수행하는 예를 제공합니다.

<br>
<br>

## 2.3 메모리 계층구조

CUDA 스레드는 [그림 6](#그림-6-메모리-계층구조)과 같이 실행 중에 여러 메모리 공간의 데이터에 액세스할 수 있습니다. 각 스레드에는 개별 로컬 메모리가 있습니다. 각 스레드 블록은 블록의 모든 스레드에 표시되는 공유 메모리를 가지며 블록과 동일한 수명을 갖습니다. 스레드 블록 클러스터의 스레드 블록은 서로의 공유 메모리에 대한 읽기, 쓰기 작업을 수행할 수 있습니다. 모든 스레드는 동일한 전역 메모리에 접할 수 있습니다.
<br>
<br>
또한 모든 스레드가 접근할 수 있는 두 개의 추가 읽기 전용 메모리인 상수 메모리 영역 및 텍스처 메모리 영역도 있습니다. 전역, 상수 및 텍스처 메모리 공간은 다양한 메모리 사용에 최적화되어 있습니다([장치 메모리 접근](#532-장치-메모리-접근) 참조). 또한 텍스처 메모리는 일부 특정 데이터 형식에 대해 다양한 주소 지정 모드와 데이터 필터링을 제공합니다 ([Texture 및 Surface 메모리](#3214-texture-및-surface-메모리) 참조).
<br>
<br>
전역, 상수 및 텍스처 메모리 공간은 동일한 애플리케이션에 의한 커널 실행 전반에 걸쳐 지속됩니다.


<p align="center">
  <img src="https://github.com/JeHeeYu/CUDA-Cpp-Programming-Guide-Korean/assets/87363461/debc2c0a-622c-47fb-8265-bd3f20e84fb9">
  <br>
  <h5 align="center">그림 6: 메모리 계층구조</h5>
  
</p>
<br>
<br>

## 2.4 이기종 프로그래밍

<p align="center">
  <img src="https://github.com/JeHeeYu/CUDA-Cpp-Programming-Guide-Korean/assets/87363461/b216c55c-d09e-48dc-955d-153f966a64c3">
  <br>
  <h5 align="center">그림 7: 이기종 프로그래밍</h5>
  
</p>
<br>
<br>


## 3. 프로그래밍 인터페이스

## 3.2.4 공유 메모리

## 3.2.5 분산 공유 메모리

## 3.2.14 Texture 및 Surface 메모리

<br>
<br>

## 5.3.2 장치 메모리 접근

## 7. C++ 언어 확장

## 8. 협력 그룹

### 8.4.1.2 클러스터 그룹

## 16.8 Compute Capability 9.0
