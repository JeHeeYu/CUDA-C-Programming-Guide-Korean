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
  - [2.3 메모리 계층구조]()
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

  <img src="https://github.com/JeHeeYu/One-Hundred-ME/assets/87363461/6a9c135c조

## 3. 프로그래밍 인터페이스

## 3.2.4 공유 메모리

## 3.2.5 분산 공유 메모리

<br>
<br>

## 7. C++ 언어 확장

## 8. 협력 그룹

### 8.4.1.2 클러스터 그룹

## 16.8 Compute Capability 9.0
