#pragma once
#ifndef LDSO_INDEX_THREAD_REDUCE_H_
#define LDSO_INDEX_THREAD_REDUCE_H_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "Settings.h"

using namespace std;
using namespace std::placeholders;

namespace ldso {

    namespace internal {

        /**
         * Multi thread tasks
         * use reduce function to multi threads a given task
         * like removing outliers or activating points
         * @tparam Running
         */
        template<typename Running>
        class IndexThreadReduce {

        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            inline IndexThreadReduce() {
                callPerIndex = bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3, _4);
                // 分配线程
                for (int i = 0; i < NUM_THREADS; i++) {
                    isDone[i] = false;
                    gotOne[i] = true;
                    workerThreads[i] = thread(&IndexThreadReduce::workerLoop, this, i);
                }

            }

            inline ~IndexThreadReduce() {
                running = false;

                exMutex.lock();
                todo_signal.notify_all();
                exMutex.unlock();

                for (int i = 0; i < NUM_THREADS; i++)
                    workerThreads[i].join();


                printf("destroyed ThreadReduce\n");

            }
            /**
            * @brief 给线程分配任务, 并判断是否完成
            * 
            * @param callPerIndex   要执行的函数, 使用bind形式传递参数
            * @param first          任务起始index
            * @param end            任务结束index
            * @param stepSize       每个线程直线的个数
            * 
            ***/
            inline void
            reduce(function<void(int, int, Running *, int)> callPerIndex, int first, int end, int stepSize = 0) {

                memset(&stats, 0, sizeof(Running));

                // 每个线程分配任务数
                if (stepSize == 0)
                    stepSize = ((end - first) + NUM_THREADS - 1) / NUM_THREADS;

                unique_lock<mutex> lock(exMutex);

                // save
                this->callPerIndex = callPerIndex;
                nextIndex = first;
                maxIndex = end;
                this->stepSize = stepSize;

                // go worker threads!
                for (int i = 0; i < NUM_THREADS; i++) {
                    // 每个线程自己工作未完成
                    isDone[i] = false;
                    // 每个线程都没执行过
                    gotOne[i] = false;  
                }

                // let them start!
                todo_signal.notify_all();


                // wait for all worker threads to signal they are done.
                while (true) {
                    // wait for at least one to finish
                    done_signal.wait(lock);

                    // check if actually all are finished.
                    bool allDone = true;
                    for (int i = 0; i < NUM_THREADS; i++)
                        allDone = allDone && isDone[i];

                    // all are finished! exit.
                    if (allDone)
                        break;
                }

                nextIndex = 0;
                maxIndex = 0;
                this->callPerIndex = bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3, _4);
            }

            Running stats;  // 所有线程计算结果求和

        private:
            thread workerThreads[NUM_THREADS];
            bool isDone[NUM_THREADS];
            bool gotOne[NUM_THREADS];

            mutex exMutex;
            condition_variable todo_signal;     // 等待任务
            condition_variable done_signal;     // 完成任务

            int nextIndex =0;
            int maxIndex =0;
            int stepSize =1;

            bool running =true;

            function<void(int, int, Running *, int)> callPerIndex;

            void callPerIndexDefault(int i, int j, Running *k, int tid) {
                printf("ERROR: should never be called....\n");
                assert(false);
            }
            /**
            * @brief 第idx个线程的执行过程
            * 
            * @param idx 线程ID号
            ***/
            void workerLoop(int idx) {
                unique_lock<mutex> lock(exMutex);

                while (running) {
                    // try to get something to do.
                    int todo = 0;
                    bool gotSomething = false;
                    // 有任务没完成, 则当前线程去做
                    if (nextIndex < maxIndex) {
                        // got something!
                        todo = nextIndex;
                        nextIndex += stepSize;
                        gotSomething = true;
                    }

                    // if got something: do it (unlock in the meantime)
                    if (gotSomething) {
                        lock.unlock();

                        assert(callPerIndex != 0);

                        Running s;
                        memset(&s, 0, sizeof(Running));
                        // 给要执行的函数传递参数
                        callPerIndex(todo, std::min(todo + stepSize, maxIndex), &s, idx); 
                        gotOne[idx] = true;
                        lock.lock();
                        // 累加
                        stats += s;
                    } // otherwise wait on signal, releasing lock in the meantime.
                    else {
                        // 如果某个线程没执行过, 执行一次
                        //? 这样做的目的是???
                        if (!gotOne[idx]) {
                            lock.unlock();
                            assert(callPerIndex != 0);
                            Running s;
                            memset(&s, 0, sizeof(Running));
                            callPerIndex(0, 0, &s, idx);  // 0到0执行
                            gotOne[idx] = true;
                            lock.lock();
                            stats += s;
                        }
                        isDone[idx] = true;
                        // 完成
                        done_signal.notify_all();
                        // 等待任务
                        todo_signal.wait(lock);
                    }
                }
            }
        };

    }
}

#endif // LDSO_INDEX_THREAD_REDUCE_H_
