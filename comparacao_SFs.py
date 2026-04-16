import numpy as np
import math
import matplotlib.pyplot as plt
import time
import gc

# Função de simulação otimizada com Mini-Batches
def simulate_lora_sf_comparison(SNR_dB, SF, num_rx, test_points, combining="mrc"):
    N_chips = 2 ** SF
    batch_size = 10000  # Tamanho seguro para RAM
    num_batches = test_points // batch_size
    total_errors = 0
    
    Pnoise_int = 10 ** ((10 * math.log10(125000) - 168) / 10)
    gamma = 10 ** (SNR_dB / 10)
    Ampl = np.sqrt(gamma * Pnoise_int)
    k = np.arange(N_chips)
    base_down_chirp = np.exp(-1j * 2 * np.pi * k * (k / N_chips))

    for _ in range(num_batches):
        symbol_indices = np.random.randint(0, N_chips, size=batch_size)
        
        # Geração LoRa
        lora_symbols = np.exp(1j * 2 * np.pi * (np.mod(symbol_indices[:, None] + k, N_chips)) * (k / N_chips))
        lora_symbols /= np.sqrt(np.mean(np.abs(lora_symbols)**2, axis=1, keepdims=True))

        # Canal Rayleigh e Ruído
        hs = (1/np.sqrt(2)) * (np.random.randn(batch_size, num_rx) + 1j * np.random.randn(batch_size, num_rx))
        noise = np.sqrt(Pnoise_int/2) * (np.random.randn(batch_size, num_rx, N_chips) + 1j * np.random.randn(batch_size, num_rx, N_chips))
        rs = hs[:, :, None] * Ampl * lora_symbols[:, None, :] + noise

        # Combinação MRC (Melhor técnica)
        numerator = np.sum(np.conjugate(hs)[:, :, None] * rs, axis=1)
        denominator = np.sum(np.abs(hs)**2, axis=1, keepdims=True)
        combined = numerator / denominator

        # Demodulação
        dechirped = combined * base_down_chirp
        fft_out = np.fft.fft(dechirped, axis=1, norm="ortho")
        dv = np.argmax(np.abs(fft_out), axis=1)

        # Cálculo de Erros de Bit
        bits_tx = ((symbol_indices[:, None] & (1 << np.arange(SF))) > 0).astype(int)
        bits_rx = ((dv[:, None] & (1 << np.arange(SF))) > 0).astype(int)
        total_errors += np.sum(bits_tx != bits_rx)
        
    return total_errors / (SF * test_points)

# --- EXECUÇÃO ---
start_time = time.time()
SNR_dBs = np.arange(-30, 12, 2)
SFs = [7, 9, 11]
L_fixo = 2 # Vamos fixar 2 antenas para ver o efeito do SF
test_points = 100000 # 100k pontos para ser rápido no seu PC

results_sf = {sf: [] for sf in SFs}

print(f"Iniciando Comparação de SF (L={L_fixo})")

for sf in SFs:
    print(f"Simulando SF {sf}...", end=" ", flush=True)
    for snr in SNR_dBs:
        ber = simulate_lora_sf_comparison(snr, sf, L_fixo, test_points)
        results_sf[sf].append(ber)
        print(".", end="", flush=True)
    print(" OK")
    gc.collect()

# --- PLOT ---
plt.figure(figsize=(10, 6))
colors_sf = {7: "orange", 9: "purple", 11: "blue"}

for sf in SFs:
    plt.semilogy(SNR_dBs, results_sf[sf], '-o', color=colors_sf[sf], label=f"SF {sf}, L={L_fixo}")

plt.grid(True, which="both", linestyle="--")
plt.xlabel("SNR (dB)", fontweight='bold')
plt.ylabel("BER", fontweight='bold')
plt.title(f"Comparação de Desempenho LoRa por SF\n(Técnica MRC, L={L_fixo})")
plt.legend()
plt.savefig("comparacao_SF_MRC.png", dpi=300)
print(f"\nSucesso! Gráfico salvo em 'comparacao_SF_MRC.png'")
plt.show()