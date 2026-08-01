from scapy.all import Ether, IP, TCP, wrpcap
import time


def generate_simulation_pcap(filename="simulasyon_trafigi.pcap"):
    packets = []

    # Zaman damgalarını (timestamp) gerçekçi tutmak için şu anki zamanı baz alıyoruz
    base_time = time.time()

    print("2. Zararlı trafik (SYN Flood DoS) oluşturuluyor...")
    # Saldırı Akışı: Aynı kaynak porttan, aynı hedefe 1 milisaniye arayla
    # gönderilen 2000 adet SYN paketi.
    #
    # ÖNEMLİ: sport SABİT tutulmalı. CICFlowMeter paketleri 5-tuple'a göre
    # (src_ip, dst_ip, src_port, dst_port, protocol) akışlara gruplar.
    # sport her paket değiştiğinde her paket ayrı bir akış (flow) olur ve
    # gerçek bir SYN flood'un yüksek paket sayısı / yüksek pkts_s imzası
    # asla oluşmaz. Bu yüzden önceki sürüm modelin normal trafik gibi
    # skorlamasına sebep oluyordu.
    attack_start_time = base_time
    fixed_sport = 4444
    packet_count = 2000

    for i in range(packet_count):
        pkt = Ether() / IP(src="10.10.10.50", dst="192.168.1.100") / TCP(
            sport=fixed_sport, dport=80, flags="S"
        )
        pkt.time = attack_start_time + (i * 0.001)  # Her paket arası 1 milisaniye (Çok hızlı)
        packets.append(pkt)

    print(f"\nToplam {len(packets)} paket bellekte oluşturuldu.")
    print(f"'{filename}' dosyasına yazılıyor...")

    # Ağa yollama (send) işlemi YAPILMAZ. Doğrudan diske dosya olarak yazılır.
    wrpcap(filename, packets)
    print("✅ İşlem başarıyla tamamlandı! Dosyayı CICFlowMeter veya Wireshark ile inceleyebilirsiniz.")


if __name__ == "__main__":
    generate_simulation_pcap()