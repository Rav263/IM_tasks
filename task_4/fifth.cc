#include <fstream>
#include <string>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/csma-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("FifthScriptExample");


class MyApp : public Application {
public:
    MyApp ();
    virtual ~MyApp();
    void Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate);

private:
    virtual void StartApplication (void);
    virtual void StopApplication (void);

    void ScheduleTx (void);
    void SendPacket (void);

    Ptr<Socket>     m_socket;
    Address         m_peer;
    uint32_t        m_packetSize;
    uint32_t        m_nPackets;
    DataRate        m_dataRate;
    EventId         m_sendEvent;
    bool            m_running;
    uint32_t        m_packetsSent;
};

MyApp::MyApp():
    m_socket (0), 
    m_peer (), 
    m_packetSize (0), 
    m_nPackets (0), 
    m_dataRate (0), 
    m_sendEvent (), 
    m_running (false), 
    m_packetsSent (0)
{}

MyApp::~MyApp() {
    m_socket = 0;
}

void MyApp::Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate) {
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_nPackets = nPackets;
    m_dataRate = dataRate;
}

void MyApp::StartApplication (void) {
    m_running = true;
    m_packetsSent = 0;
    m_socket->Bind ();
    m_socket->Connect (m_peer);
    SendPacket ();
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

    if (++m_packetsSent < m_nPackets) {
        ScheduleTx ();
    }
}

void MyApp::ScheduleTx (void) {
    if (m_running) {
        //Time tNext (Seconds (m_packetSize * 8 / static_cast<double> (m_dataRate.GetBitRate ())));
        Time tNext(NanoSeconds(10000));
        m_sendEvent = Simulator::Schedule (tNext, &MyApp::SendPacket, this);
    }
}


static void RxDrop (std::string context, Ptr<const Packet> p) {
    NS_LOG_UNCOND (context << " RxDrop at " << Simulator::Now ().GetSeconds ());
}

/*static void TxBegin (std::string context, Ptr<const Packet> p) {
    NS_LOG_UNCOND (context << "TxBegin at " << Simulator::Now ().GetSeconds ());
}*/

static void QueDrop (std::string context, Ptr<const Packet> p) {
    NS_LOG_UNCOND (context << " Queue Drop at " << Simulator::Now ().GetSeconds ());
}

static void MacTxDrop (std::string context, Ptr<const Packet> p) {
    NS_LOG_UNCOND (context << "MacTxDrop at " << Simulator::Now ().GetSeconds ());
}
uint64_t csma_num = 2;

int main (int argc, char *argv[]) {
    CommandLine cmd;
    cmd.Parse (argc, argv);
  
    NodeContainer nodes;
    nodes.Create (csma_num);
    
    CsmaHelper csma;
    csma.SetChannelAttribute ("DataRate", StringValue ("1Mbps"));
    csma.SetChannelAttribute ("Delay", TimeValue (Seconds(1)));
    csma.SetQueue("ns3::DropTailQueue");

    NetDeviceContainer devices = csma.Install (nodes);

    for (uint32_t i = 0; i < csma_num; i++) {
        Ptr<RateErrorModel> em = CreateObject<RateErrorModel> ();
        em->SetAttribute ("ErrorRate", DoubleValue (0.0001));
        Ptr<DropTailQueue<Packet>> que = CreateObject<DropTailQueue<Packet>>();
        que->SetMaxSize(QueueSize("2p"));
        que->TraceConnect("Drop", "Host " + std::to_string(i) + ": ", MakeCallback(&QueDrop));
        devices.Get(i)->SetAttribute("TxQueue", PointerValue(que));
        devices.Get(i)->SetAttribute("ReceiveErrorModel", PointerValue (em));
    }
    InternetStackHelper stack;
    stack.Install (nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign (devices);

    uint16_t sinkPort = 8080;
    Address sinkAddress (InetSocketAddress (interfaces.GetAddress (csma_num - 1), sinkPort));
    PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
    ApplicationContainer sinkApps = packetSinkHelper.Install (nodes.Get (csma_num - 1));
    sinkApps.Start (Seconds (0.));
    sinkApps.Stop (Seconds (20.));


    for (uint32_t i = 0; i < csma_num - 1; i++) {
        Ptr<Socket> ns3TcpSocket = Socket::CreateSocket (nodes.Get (i), UdpSocketFactory::GetTypeId ());
        //`ns3TcpSocket->TraceConnectWithoutContext ("CongestionWindow", MakeCallback (&CwndChange));
        Ptr<MyApp> app = CreateObject<MyApp> ();
        app->Setup (ns3TcpSocket, sinkAddress, 1040, 100000, DataRate ("100Mbps"));
        nodes.Get (i)->AddApplication (app);
        app->SetStartTime (Seconds (1.));
        app->SetStopTime (Seconds (20.));
//        devices.Get (i)->TraceConnect("PhyTxBegin","Host " + std::to_string(i) + ": ", MakeCallback (&TxBegin));
        devices.Get (i)->TraceConnect("MacTxDrop","Host " + std::to_string(i) + ": ", MakeCallback (&MacTxDrop));
    }

    devices.Get (csma_num - 1)->TraceConnect("PhyRxDrop", "Server:" , MakeCallback (&RxDrop));

    //std::ofstream file("fifth.tr", std::ios_base::binary | std::ios_base::out);

    AsciiTraceHelper ascii;
    csma.EnableAsciiAll (ascii.CreateFileStream ("fifth.tr"));


    Simulator::Stop (Seconds (20));
    Simulator::Run ();
    Simulator::Destroy ();

    //file.close();
    return 0;
}

