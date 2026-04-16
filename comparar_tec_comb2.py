import numpy as np
import math
import matplotlib.pyplot as plt
import time
import gc

# -------------------------------------------------------------
# Função auxiliar (Otimizada para processar vários símbolos de uma vez)
def symbols_to_bits(symbols, SF):
    # Converte um array de símbolos em uma matriz de bits
    bits = ((symbols[:, None] & (1 << np.arange(SF))) > 0).astype(int)
    return np.flip(bits, axis=1)

# -------------------------------------------------------------
# Simulação LoRa com Mini-Batches (Vetorizada)
def simulate_lora_diversity(SNR_dB, SF, N_chips, base_down_chirp,
                            num_rx, test_points, combining="mrc"):

    gamma = 10 ** (SNR_dB / 10)
    Ampl = np.sqrt(gamma * Pnoise)
    
    # Configuração do Mini-Batch (Ajustado para seus 6.7GB de RAM)
    batch_size = 20000 
    num_batches = test_points // batch_size
    total_errors = 0
    k = np.arange(N_chips)

    for _ in range(num_batches):
        # 1. Gera um bloco de índices de símbolos aleatórios
        symbol_indices = np.random.randint(0, N_chips, size=batch_size)

        # 2. Geração vetorial do símbolo LoRa (Matriz: batch_size x N_chips)
        lora_symbols = np.exp(
            1j * 2 * np.pi * np.mod(symbol_indices[:, None] + k, N_chips) * (k / N_chips)
        )
        # Normalização de energia
        lora_symbols /= np.sqrt(np.mean(np.abs(lora_symbols) ** 2, axis=1, keepdims=True))

        # 3. Canal Rayleigh (Matriz: batch_size x num_rx)
        hs = (1 / np.sqrt(2)) * (
            np.random.randn(batch_size, num_rx) + 1j * np.random.randn(batch_size, num_rx)
        )

        # 4. Ruído AWGN (Matriz: batch_size x num_rx x N_chips)
        noise = np.sqrt(Pnoise / 2) * (
            np.random.randn(batch_size, num_rx, N_chips) + 1j * np.random.randn(batch_size, num_rx, N_chips)
        )

        # 5. Sinal Recebido (Broadcast das dimensões para bater)
        # rs shape: (batch_size, num_rx, N_chips)
        rs = hs[:, :, None] * Ampl * lora_symbols[:, None, :] + noise

        # ------------------ Combinação Vetorizada ------------------
        if combining == "mrc":
            num = np.sum(np.conjugate(hs)[:, :, None] * rs, axis=1)
            den = np.sum(np.abs(hs) ** 2, axis=1, keepdims=True)
            combined_signal = num / den

        elif combining == "egc":
            phase = hs / (np.abs(hs) + 1e-12)
            combined_signal = np.sum(np.conjugate(phase)[:, :, None] * rs, axis=1) / num_rx

        elif combining == "sc":
            idx = np.argmax(np.abs(hs), axis=1)
            combined_signal = rs[np.arange(batch_size), idx]

        # 6. Demodulação (FFT no bloco inteiro)
        dechirped = combined_signal * base_down_chirp
        combined_freq = np.abs(np.fft.fft(dechirped, axis=1, norm="ortho"))
        dv = np.argmax(combined_freq, axis=1)

        # 7. Contagem de erros (Comparando matrizes de bits)
        bits_tx = symbols_to_bits(symbol_indices, SF)
        bits_rx = symbols_to_bits(dv, SF)
        total_errors += np.sum(bits_tx != bits_rx)
        
        # Limpeza de memória para o próximo batch
        del lora_symbols, hs, noise, rs, combined_signal, combined_freq
        
    return total_errors / (SF * test_points)

# -------------------------------------------------------------
# Parâmetros
SF = 10
BW = 125000
Pnoise = 10 ** ((10 * math.log10(BW) - 168) / 10)

N_chips = 2 ** SF
k = np.arange(N_chips)
base_down_chirp = np.exp(-1j * 2 * np.pi * k * (k / N_chips))

SNR_dBs = np.arange(-30, 12, 2)
test_points = 200000
Ls = [1, 2, 3, 4, 5]

# -------------------------------------------------------------
# Simulação
results = {c: {L: [] for L in Ls} for c in ["sc", "egc", "mrc"]}
start_time = time.time()

for combining in results:
    print(f"\n=== Técnica: {combining.upper()} ===")
    for L in Ls:
        print(f"L={L}: ", end="", flush=True)
        for snr in SNR_dBs:
            ber = simulate_lora_diversity(
                snr, SF, N_chips, base_down_chirp,
                L, test_points, combining=combining
            )
            results[combining][L].append(ber)
            print(".", end="", flush=True)
        print(" OK")
        gc.collect()

# -------------------------------------------------------------
# Plot (Mantendo o seu estilo original)
plt.figure(figsize=(10, 5))
combining_colors = {"sc": "green", "egc": "blue", "mrc": "red"}
line_styles = {1: "-", 2: "--", 3: ":", 4: "-o", 5: "-x"}

for combining, color in combining_colors.items():
    for L in Ls:
        plt.semilogy(
            SNR_dBs, results[combining][L],
            line_styles[L], color=color, linewidth=2,
            label=f"{combining.upper()}, L={L}"
        )

plt.grid(True, which="both", linestyle="--", linewidth=0.6)
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title(f"BER × SNR (SF={SF}) - Mini-Batches Otimizado\nSC vs EGC vs MRC (SISO → SIMO 1×5)")
plt.ylim(1e-5, 1)
plt.legend(ncol=3, fontsize=9)
plt.tight_layout()

plt.savefig("ber_sc_egc_mrc_simo.png", dpi=300)
print(f"\n✅ Concluído em {(time.time()-start_time)/60:.2f} min. Gráfico salvo!")
plt.show()