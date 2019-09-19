#include <iostream>
#include <list>
#include <vector>
#include <thread>

#define EV_INIT 1
#define EV_REQ 2
#define EV_FIN 3

#define RUN 1
#define IDLE 0

#define LIMIT 100


class Event {
    float time;
    int type;
    int attr;
public:
    Event(float time, int type, int attr): 
        time(time), type(type), attr(attr){}
    
    float get_time() {
        return time;
    }
};


class Request {
    float time;
    int source_num;
public:
    Request(float time, int source_num):
        time(time), source_num(source_num){}
};


class Calendar: public std::list<Event *> {
public:
    void put(Event *);
    Event* get();
};


class Supervisor {
    
};

class Server {

};


void Calendar::put(Event *event) {
    auto iter = begin();
    for (;iter != end() && (*iter)->get_time() <= event->get_time(); ++iter);
    
    insert(iter, event);
}

Event *Calendar::get() {
    if (empty())
        return nullptr;

    Event *tmp_event = front();
    pop_front();

    return tmp_event;
}

int main() {
    return 0;
}
