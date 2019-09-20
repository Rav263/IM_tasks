#include <iostream>
#include <list>
#include <vector>
#include <thread>
#include <chrono>
#include <functional>


#define EV_INIT 1
#define EV_REQ 2
#define EV_FIN 3
#define EV_END_TIME 4

#define RUN 1
#define IDLE 0

#define LIMIT 100


class Event {
    float time;
    int type;
    int server_id;
    int attr;
public:
    Event(float time, int type, int attr, int server_id): 
        time(time), type(type), attr(attr), server_id(server_id) {}
   
    void print() {
        std::cout << "time: "      << time      << " "
                  << "type: "      << type      << " "
                  << "server_id: " << server_id << " "
                  << "attr: "      << attr      << std::endl;
    }
    int get_attr() {
        return attr;
    }

    float get_time() { 
        return time; 
    }

    int get_type() {
        return type;
    }

    int get_server_id(){
        return server_id;
    }
};


class Request {
    float time;
    int source_num;
public:
    Request(float time, int source_num):
        time(time), source_num(source_num){}

    float get_time() {
        return time;
    }

    int get_source_num() {
        return source_num;
    }
};


class Calendar: public std::list<Event *> {
public:
    void put(Event *);
    Event* get(int server_id);
};

class Times {
    float current_time;
    float run_begin_time;
public:
    Times() {
        this->current_time = 0;
        this->run_begin_time = 0;
    }

    Times(float current_time, float run_begin_time):
        current_time(current_time), run_begin_time(run_begin_time) {}

    float get_run_time() {
        return run_begin_time;
    }

    float get_current_time() {
        return current_time;
    }

    void set_run_time(float run_begin_time) {
        this->run_begin_time = run_begin_time;
    }

    void set_current_time(float current_time) {
        this->current_time = current_time;
    }
};


class Supervisor {
    Times times[2];
    Calendar *calendar;
public:
    Supervisor(Calendar *calendar): calendar(calendar) {}

    Times *get_time(int server_id) {
        return &times[server_id];
    }

    Event *get_event(int server_id) {
        // берём ближайшее для сервера событие или ближайшее событие без индификатора сервера
        Event *current_event = calendar->get(server_id);

        if (current_event == nullptr) {
            return nullptr;
        }

        times[server_id].set_current_time(current_event->get_time());
        if (times[server_id].get_current_time() > LIMIT) {
            return new Event(0, EV_END_TIME, 0, server_id);
        }

        return current_event;
    }

    void add_event(float event_time, int server_id, int event_type, int attr) {
        int id = event_type == EV_REQ ? -1 : server_id;

        if (event_type == EV_FIN) {
            times[server_id].set_run_time(times[server_id].get_current_time());
            
            std::cout << times[server_id].get_current_time() << " " << event_time << std::endl;
        }

        calendar->put(new Event(times[server_id].get_current_time() + event_time,
                                event_type, attr, id));
    }
};

typedef std::list<Request*> Queue;

class Server {
    Supervisor *supervisor;
    int cpu_state;
    int server_id;
    int request_num = 0;
    int pause_num = 0;


public:
    Server(int server_id, Supervisor *supervisor, int cpu_state): 
        server_id(server_id), supervisor(supervisor), cpu_state(cpu_state) {}


    float get_req_time(int server_id, int source_num) {
        float request = ((float) rand()) / RAND_MAX;
        
        std::cout << "Server id: " << server_id << " "
                  << "Request num: " << request_num << std::endl;
   
        request_num += 1;

        return source_num == 1 ? request * 10 : request * 20;
    }

    float get_pause_time(int server_id, int source_num) {
        float pause = ((float) rand()) / RAND_MAX;

        std::cout << "Server id: " << server_id << " "
                  << "Pause num: " << pause_num << std::endl;

        pause_num += 1;

        return source_num == 1 ? pause * 20 : pause * 10;
    }

    void run_server() {
        Queue queue;
        
        while (true) {
            Event *current_event = supervisor->get_event(server_id);


            if (current_event == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            current_event->print();

            if (current_event->get_type() == EV_INIT) {
                supervisor->add_event(0, server_id, EV_REQ, 1);
                supervisor->add_event(0, server_id, EV_REQ, 2);

            } else if (current_event->get_type() == EV_REQ) {
                float event_time = get_req_time(server_id, current_event->get_attr());
                
                std::cout << "server: " << server_id 
                          << " dt: "   << event_time
                          << " num: "  << current_event->get_attr() << std::endl;
                

                if (cpu_state == IDLE) {
                    cpu_state = RUN;
                    supervisor->add_event(event_time, server_id, EV_FIN, 
                                          current_event->get_attr());
                } else {
                    queue.push_back(new Request(event_time, current_event->get_attr()));
                }

                supervisor->add_event(get_pause_time(server_id, current_event->get_attr()), server_id,
                                                     EV_REQ, current_event->get_attr());
            } else if (current_event->get_type() == EV_FIN) {
                cpu_state = IDLE;
                Times *time = supervisor->get_time(server_id);

                std::cout << "server: "    << server_id 
                          << " Работа с: " << time->get_run_time()
                          << " по: "       << time->get_current_time()
                          << " длит.: "    << (time->get_current_time() - 
                                               time->get_run_time()) << std::endl; 
             

                if (!queue.empty()) {
                    cpu_state = RUN;
                    std::cout << "GET EVENT FROM QUEUE:: server_id: " << server_id << std::endl;

                    Request *request = queue.front();
                    queue.pop_front();
                    float tt = request->get_time();
        
                    supervisor->add_event(tt, server_id,
                                          EV_FIN, request->get_source_num());
                    
                    delete request;
                }
            } else if (current_event->get_type() == EV_END_TIME) {
                delete current_event;
                break;
            }

            delete current_event;
        }
    }
};


void Calendar::put(Event *event) {
    auto iter = begin();
    for (;iter != end() && (*iter)->get_time() <= event->get_time(); ++iter);
    
    insert(iter, event);
}

Event *Calendar::get(int server_id) {
    if (empty())
        return nullptr;

    auto iter = begin();

    for (;iter != end() && (*iter)->get_server_id() != server_id && (*iter)->get_server_id() != -1; ++iter) {
    }
    if (iter == end())
        return nullptr;

    Event *tmp_event = *iter;
    erase(iter);

    return tmp_event;
}

int main() {
    Calendar *calendar = new Calendar();
    Supervisor *supervisor = new Supervisor(calendar);
    srand(2019);

    supervisor->add_event(0, 0, EV_INIT, 0);
    supervisor->add_event(0, 1, EV_INIT, 0);
    Server *first_server  = new Server(0, supervisor, IDLE);
    Server *second_server = new Server(1, supervisor, IDLE);



    std::thread server_1(std::bind(&Server::run_server, first_server));
    std::thread server_2(std::bind(&Server::run_server, second_server));


    server_2.join();
    server_1.join();


    delete calendar;
    delete supervisor;
    delete first_server;
    delete second_server;


    return 0;
}
