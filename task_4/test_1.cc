#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <random>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/csma-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("MyApp");

using gener = std::default_random_engine;
using distr = std::exponential_distribution<double>;

double ln2 = 0.69314718056;
uint64_t total_sended = 0;
uint64_t total_droped = 0;
uint64_t total_ended_way = 0;
uint64_t back_off_total = 0;
double queue_size_midle = 0;

//#define LOG
//#define DROP_LOG
//#define RX_BEGIN_LOG
//#define BACKOFF_LOG

class MyApp : public Application {
public:
    MyApp ();
    virtual ~MyApp();
    void Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, 
                gener *generator, distr *distribution, Ptr<Queue<Packet>> queue,
                std::string host_name);

private:
    virtual void StartApplication (void);
    virtual void StopApplication (void);

    void ScheduleTx (void);
    void SendPacket (void);
    void PrintLog (void);

    Ptr<Socket>     m_socket;
    Address         m_peer;
    uint32_t        m_packetSize;
    EventId         m_sendEvent;
    EventId         m_timeEvent;
    bool            m_running;
    uint32_t        m_packetsSent;
    gener          *m_generator;
    distr          *m_distribution;
    Ptr<Queue<Packet>>      m_queue;
    std::string     m_name;
};

MyApp::MyApp():
    m_socket (0), 
    m_peer (), 
    m_packetSize (0), 
    m_sendEvent (), 
    m_running (false), 
    m_packetsSent (0)
{}

MyApp::~MyApp() {
    m_socket = 0;
    delete m_generator;
    delete m_distribution;
}

void MyApp::Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, 
                   gener *generator, distr *distribution, Ptr<Queue<Packet>> queue,
                   std::string host_name) {
    m_socket       = socket;
    m_peer         = address;
    m_packetSize   = packetSize;
    m_generator    = generator;
    m_distribution = distribution;
    m_queue        = queue;
    m_name         = host_name;
}

void MyApp::StartApplication (void) {
    m_running = true;
    m_packetsSent = 0;
    
    m_socket->Bind ();
    m_socket->Connect (m_peer);
    m_socket->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>());
    m_socket->SetAllowBroadcast(true);

    m_sendEvent = Simulator::Schedule(Seconds(0.0), &MyApp::SendPacket, this);
    m_timeEvent = Simulator::Schedule(Seconds(0.0), &MyApp::PrintLog, this);
}

void MyApp::PrintLog(void) {
    if (m_name == "Host 0")
        NS_LOG_INFO(m_name << ": Current time " << (Simulator::Now()).GetSeconds());
    m_timeEvent = Simulator::Schedule(Seconds(0.5), &MyApp::PrintLog, this);
}


void MyApp::StopApplication (void) {
    m_running = false;

    if (m_sendEvent.IsRunning ()) {
        Simulator::Cancel (m_sendEvent);
    }

    if (m_socket) {
        m_socket->Close ();
    }
}

void MyApp::SendPacket (void) {
    Ptr<Packet> packet = Create<Packet> (m_packetSize);
    m_socket->Send (packet);

    ++total_sended;
#ifdef LOG
    NS_LOG_INFO (m_name << ": TraceDelay TX " << m_packetSize << " bytes to " << " Uid: "
                                              << packet->GetUid () << " Time: "
                                              << (Simulator::Now ()).GetSeconds ());
#endif
    ScheduleTx ();
}

void MyApp::ScheduleTx (void) {
    if (m_running) {
        Time tNext(MilliSeconds((*m_distribution)(*m_generator)));
        queue_size_midle = ((queue_size_midle * total_sended) +  m_queue->GetNPackets()) / (total_sended + 1);
#ifdef LOG
        NS_LOG_INFO (m_name << ": Time: " << tNext << " QUEUE SIZE: " << m_queue->GetNPackets());
#endif
        m_sendEvent = Simulator::Schedule (tNext, &MyApp::SendPacket, this);
    }
}

static void RxBegin (std::string context, Ptr<const Packet> p) {
++total_ended_way;
#ifdef RX_BEGIN_LOG
    NS_LOG_UNCOND (context << " RxBegin at " << Simulator::Now ().GetSeconds ());
#endif
}

static void QueDrop (std::string context, Ptr<const Packet> p) {
    ++total_droped;
#ifdef DORP_LOG
    NS_LOG_UNCOND (context << "Queue Drop at " << Simulator::Now ().GetSeconds ());
#endif
}

static void MacTxDrop (std::string context, Ptr<const Packet> p) {
#ifdef LOG
    NS_LOG_UNCOND (context << "MacTxDrop at " << Simulator::Now ().GetSeconds ());
#endif
}

static void MacTxBackoff (std::string context, Ptr<const Packet> p) {
++back_off_total;
#ifdef BACKOFF_LOG
    NS_LOG_UNCOND (context << "Backoff at " << Simulator::Now ().GetSeconds ());
#endif
}


uint64_t csma_num = 10;
double dist = 0.1;
uint64_t channel_delay = 300;


int main (int argc, char *argv[]) {
    CommandLine cmd;
    
    cmd.AddValue("hosts", "Number of hosts in simulation", csma_num);
    cmd.AddValue("distr", "Distribution parametr", dist);
    cmd.AddValue("delay", "Delay in channel", channel_delay);
    cmd.Parse (argc, argv);

    LogComponentEnable ("MyApp", LOG_LEVEL_INFO);
    csma_num += 1;

    NodeContainer nodes;
    nodes.Create (csma_num);
    
    CsmaHelper csma;
    csma.SetChannelAttribute ("DataRate", StringValue ("1000Mbps"));
    csma.SetChannelAttribute ("Delay", TimeValue (NanoSeconds(channel_delay)));//1982)));
    csma.SetQueue ("ns3::DropTailQueue");

    NetDeviceContainer devices = csma.Install (nodes);

    std::vector<Ptr<Queue<Packet>>> queues;
    for (uint32_t i = 0; i < csma_num - 1; i++) {
        Ptr<RateErrorModel> em = CreateObject<RateErrorModel> ();
        em->SetAttribute ("ErrorRate", DoubleValue (0.00000001));
        Ptr<DropTailQueue<Packet>> que = CreateObject<DropTailQueue<Packet>>();
        que->SetMaxSize(QueueSize("100p"));
        que->TraceConnect("Drop", "Host " + std::to_string(i) + ": ", MakeCallback(&QueDrop));
        queues.push_back(que);
        devices.Get(i)->SetAttribute("TxQueue", PointerValue(que));
        devices.Get(i)->SetAttribute("ReceiveErrorModel", PointerValue (em));
    }
    InternetStackHelper stack;
    stack.Install (nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.0.0", "255.255.0.0");
    Ipv4InterfaceContainer interfaces = address.Assign (devices);

    uint16_t sinkPort = 8080;
    Address sinkAddress (InetSocketAddress (interfaces.GetAddress (csma_num - 1), sinkPort));
    PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
    ApplicationContainer sinkApps = packetSinkHelper.Install (nodes.Get (csma_num - 1));
    sinkApps.Start (Seconds (0.));
    sinkApps.Stop (Seconds (10.));


    for (uint32_t i = 0; i < csma_num - 1; i++) {
        Ptr<Socket> ns3UdpSocket = Socket::CreateSocket (nodes.Get (i), UdpSocketFactory::GetTypeId ());
        Ptr<MyApp> app = CreateObject<MyApp> ();
        app->Setup (ns3UdpSocket, sinkAddress, 1500, new gener(i), new distr(dist), queues[i], "Host " + std::to_string(i));
        nodes.Get (i)->AddApplication (app);
        app->SetStartTime (Seconds (0.));
        app->SetStopTime (Seconds (10.));
        devices.Get (i)->TraceConnect("MacTxDrop","Host " + std::to_string(i) + ": ", MakeCallback (&MacTxDrop));
        devices.Get (i)->TraceConnect("MacTxBackoff", "Host " + std::to_string(i) + ": ", MakeCallback (&MacTxBackoff));
    }

    devices.Get (csma_num - 1)->TraceConnect("MacRx", "Server:" , MakeCallback (&RxBegin));
    AsciiTraceHelper ascii;
    csma.EnableAsciiAll (ascii.CreateFileStream ("fifth.tr"));
    csma.EnablePcap("test", devices.Get(csma_num - 1), true);

    Simulator::Stop (Seconds (10));
    Simulator::Run ();
    Simulator::Destroy ();

    double backoff_midle = (double) back_off_total / (total_sended - total_droped);

    std::cout << "Total sended: " << total_sended << std::endl 
              << "Total droped: " << total_droped << std::endl
              << "Queue size midle: " << queue_size_midle << std::endl
              << "Getted from Server: " << total_ended_way << std::endl
              << "Backoff times: " << back_off_total << std::endl
              << "Backoff times for one packet: " << backoff_midle << std::endl;
    return 0;
}

