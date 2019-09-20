#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <list>
#include <iostream>

using namespace std;

// Очень простой пример построения имитационной модели с календарём событий 
// Модель "самодостаточная" - не используются библиотеки для организации имитационного моделирования
// Возможности С++ используются недостаточно. Что можно улучшить в этой части?

class Event {  // событие в календаре
public:
    float time; // время свершения события
    int type;   // тип события
    int attr; // дополнительные сведения о событии в зависимости от типа
    int server_id;
    Event(float t, int tt, int a, int server_id) {
        time = t; 
        type = tt; 
        attr = a;
        this->server_id = server_id;
    } 
};


// типы событий
#define EV_INIT 1
#define EV_REQ 2
#define EV_FIN 3


// состояния
#define RUN 1
#define IDLE 0
#define LIMIT 100 // время окончания моделирования



class Request { // задание в очереди
public:
    float time;     // время выполнения задания без прерываний 
    int source_num; // номер источника заданий (1 или 2)
    Request(float t, int s) {time = t; source_num = s;} 
};  



class Calendar: public list<Event*> { // календарь событий
public:
    void put (Event* ev); // вставить событие в список с упорядочением по полю time
    Event* get (); // извлечь первое событие из календаря (с наименьшим модельным временем)
};




void Calendar::put(Event *ev) {
    Event **e = new (Event*);
    *e = ev;
    
    if(empty()) {
        push_back(*e); 
        return; 
    }

    list<Event*>::iterator i = begin();
    
    while((i != end()) && ((*i)->time <= ev->time)) 
    {
        ++i;
    }
    
    insert(i, *e);
} 



Event* Calendar::get()
{  
  if(empty()) 
      return NULL; 
  
  Event *e = front(); 
  pop_front();
  
  return e;
}


typedef list<Request*> Queue; // очередь заданий к процессору 

float get_req_time(int source_num); // длительность задания
float get_pause_time(int source_num); // длительность паузы между заданиями


int main(int argc, char **argv ) {
    Calendar calendar;
    Queue queue_1;
    Queue queue_2;

    float curr_time_1 = 0;
    float curr_time_2 = 0;

    Event *curr_ev;

    float dt;
    int cpu_state_1 = IDLE;
    int cpu_state_2 = IDLE;
    float run_begin_1;
    float run_begin_2;

    srand(2019);
    // начальное событие и инициализация календаря
    

    curr_ev = new Event(curr_time_1, EV_INIT, 0, 0);
    calendar.put( curr_ev );
    
    // цикл по событиям
    while ((curr_ev = calendar.get()) != NULL) {
        cout << " time "   << curr_ev->time 
             << " type "   << curr_ev->type 
             << " server " << curr_ev->server_id << endl;
       
        //curr_time = curr_ev->time; // продвигаем время
        
        // обработка события
        if (curr_time_1 >= LIMIT && curr_time_2 >= LIMIT)
            break; // типичное дополнительное условие останова моделирования
        

        switch(curr_ev->type) {
        case EV_INIT:  // запускаем генераторы запросов
            calendar.put(new Event(curr_time_1, EV_REQ, 1, 0));  
            calendar.put(new Event(curr_time_1, EV_REQ, 2, 0));  
            
            break;
        case EV_REQ:
            // планируем событие окончания обработки, если процессор свободен, иначе ставим в очередь
            
            dt = get_req_time(curr_ev->attr); 
	        cout << "dt "   << dt 
                 << " num " << curr_ev->attr << endl;

            if (curr_time_1 <= curr_time_2) {
                curr_time_1 = curr_ev->time;
            
                if(cpu_state_1 == IDLE) { 
	                cpu_state_1 = RUN; 
	            
                    calendar.put(new Event(curr_time_1 + dt, EV_FIN, curr_ev->attr, 1)); 
	                run_begin_1 = curr_time_1;
	        
                } else queue_1.push_back(new Request(dt, curr_ev->attr));  
  
                // планируем событие генерации следующего задания
                calendar.put(new Event(curr_time_1 + get_pause_time(curr_ev->attr), EV_REQ, curr_ev->attr, 0)); 
            } else {
                curr_time_2 = curr_ev->time;
            
                if(cpu_state_2 == IDLE) { 
	                cpu_state_2 = RUN; 
	            
                    calendar.put(new Event(curr_time_2 + dt, EV_FIN, curr_ev->attr, 2)); 
	                run_begin_2 = curr_time_2;
	        
                } else queue_2.push_back(new Request(dt, curr_ev->attr));  
  
                // планируем событие генерации следующего задания
                calendar.put(new Event(curr_time_2 + get_pause_time(curr_ev->attr), EV_REQ, curr_ev->attr, 0)); 
            }
            break;
        case EV_FIN:
            if (curr_ev->server_id == 1) {
                curr_time_1 = curr_ev->time;
                // объявляем процессор свободным и размещаем задание из очереди, если таковое есть
                cpu_state_1 = IDLE; 
            
                // выводим запись о рабочем интервале
                cout << "Server: 1 " <<
                        "Работа с "  << run_begin_1 << 
                        " по "       << curr_time_1 << 
                        " длит. "    << (curr_time_1-run_begin_1) << endl; 
            
                if (!queue_1.empty()) {
                    std::cout << "event from 1 server queue" << std::endl;
                    cpu_state_1 = RUN;
	            
                    Request *rq = queue_1.front(); 
	                queue_1.pop_front(); 
	                calendar.put(new Event(curr_time_1 + rq->time, EV_FIN, rq->source_num, 1)); 
	            
                    delete rq; 
	                run_begin_1 = curr_time_1;
	            } break;
            } else {
                curr_time_2 = curr_ev->time;
                // объявляем процессор свободным и размещаем задание из очереди, если таковое есть
                cpu_state_2 = IDLE; 
            
                // выводим запись о рабочем интервале
                cout << "Server: 2 " <<
                        "Работа с "  << run_begin_2 << 
                        " по "       << curr_time_2 << 
                        " длит. "    << (curr_time_2 - run_begin_2) << endl; 
            
                if (!queue_2.empty()) {
                    std::cout << "event from 2 server queue" << std::endl;
                    cpu_state_2 = RUN;
	            
                    Request *rq = queue_2.front(); 
	                queue_2.pop_front(); 
	                calendar.put(new Event(curr_time_2 + rq->time, EV_FIN, rq->source_num, 1)); 
	            
                    delete rq; 
	                run_begin_2 = curr_time_2;
	            } break;
            }
        } // switch
        delete curr_ev;
    } // while
} // main

int rc = 0; int pc = 0;
float get_req_time(int source_num) {
// Для демонстрационных целей - выдаётся случайное значение
// при детализации модели функцию можно доработать
   double r = ((double)rand())/RAND_MAX;
   cout << "req " << rc << endl; rc++;
   if (source_num == 1)
       return r * 10;
   else
       return r * 20; 
}

float get_pause_time(int source_num) // длительность паузы между заданиями
{  
// см. комментарий выше
   double p = ((double)rand())/RAND_MAX;
   cout << "pause " << pc << endl; pc++;
   if (source_num == 1) 
       return p * 20;
   else 
       return p * 10; 
}

