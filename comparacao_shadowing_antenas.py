import numpy as np
import math
import matplotlib.pyplot as plt
import time
import gc

def simulate_lora_shadowing(SNR_dB, SF, num_rx, test_points, sigma_shadow=0):
    N_chips = 2 ** SF
    batch_size = 10000 
    num_batches = test_points // batch_size
    total_errors = 0
    
    Pnoise_int = 10 ** ((10 * math.log10(125000) - 168) / 10)
    gamma = 10 ** (SNR_dB / 10)
    Ampl = np.sqrt(gamma * Pnoise_int)
    k = np.arange(N_chips)
    base_down_chirp = np.exp(-1j * 2 * np.pi * k * (k / N_chips))

    for _ in range(num_batches):
        symbol_indices = np.random.randint(0, N_chips, size=batch_size)
        lora_symbols = np.exp(1j * 2 * np.pi * (np.mod(symbol_indices[:, None] + k, N_chips)) * (k / N_chips))
        lora_symbols /= np.sqrt(np.mean(np.abs(lora_symbols)**2, axis=1, keepdims=True))

        # 1. Gerando o Shadowing (Lognormal Fading)
        if sigma_shadow > 0:
            s_db = np.random.normal(0, sigma_shadow, size=(batch_size, 1))
            shadow_gain = 10 ** (s_db / 20) 
        else:
            shadow_gain = 1

        # 2. Canal Rayleigh (Fast Fading)
        hs = (1/np.sqrt(2)) * (np.random.randn(batch_size, num_rx) + 1j * np.random.randn(batch_size, num_rx))
        
        # O canal total é a combinação dos dois:
        h_total = hs * shadow_gain

        noise = np.sqrt(Pnoise_int/2) * (np.random.randn(batch_size, num_rx, N_chips) + 1j * np.random.randn(batch_size, num_rx, N_chips))
        rs = h_total[:, :, None] * Ampl * lora_symbols[:, None, :] + noise

        # MRC - Maximal Ratio Combining 
        num = np.sum(np.conjugate(h_total)[:, :, None] * rs, axis=1)
        den = np.sum(np.abs(h_total)**2, axis=1, keepdims=True)
        combined = num / den

        dechirped = combined * base_down_chirp
        fft_out = np.fft.fft(dechirped, axis=1, norm="ortho")
        dv = np.argmax(np.abs(fft_out), axis=1)

        bits_tx = ((symbol_indices[:, None] & (1 << np.arange(SF))) > 0).astype(int)
        bits_rx = ((dv[:, None] & (1 << np.arange(SF))) > 0).astype(int)
        total_errors += np.sum(bits_tx != bits_rx)
        
        # Limpeza do batch
        del lora_symbols, hs, h_total, noise, rs, combined, dechirped, fft_out
        
    return total_errors / (SF * test_points)

# --- EXECUÇÃO ---
SF = 10
test_points = 100000
SNR_dBs = np.arange(-20, 15, 2) 

# Adicionando a lista de antenas
Ls = [1, 2, 4]

print(f"Iniciando Simulação de Shadowing com Diversidade (SF={SF})...")

# Dicionários para guardar os resultados de cada L
res_no_shadow = {L: [] for L in Ls}
res_heavy_shadow = {L: [] for L in Ls}

for L in Ls:
    print(f"\n--- Simulando para L={L} Antena(s) ---")
    
    print("Cenário Céu Aberto (Shadowing 0dB)...", end=" ", flush=True)
    for snr in SNR_dBs:
        res_no_shadow[L].append(simulate_lora_shadowing(snr, SF, L, test_points, 0))
        print(".", end="", flush=True)

    print("\nCenário Bloqueio Severo (Shadowing 6dB)...", end=" ", flush=True)
    for snr in SNR_dBs:
        res_heavy_shadow[L].append(simulate_lora_shadowing(snr, SF, L, test_points, 6))
        print(".", end="", flush=True)
    
    gc.collect()

# --- SALVAMENTO DOS DADOS ---
dados_para_salvar = {
    'snr': SNR_dBs,
    'ber_no_shadow': res_no_shadow,
    'ber_heavy_shadow': res_heavy_shadow
}
# Usando np.save para facilitar a leitura de dicionários aninhados depois
np.save("dados_shadowing.npy", dados_para_salvar)
print(f"\n\nDados salvos em 'dados_shadowing.npy'")


# --- PLOT ---
plt.figure(figsize=(11, 7))

# Mapeamento de cores para manter o padrão visual
color_map = {1: 'black', 2: 'blue', 4: 'red'}

for L in Ls:
    c = color_map[L]
    # Céu aberto (linha sólida, círculo)
    plt.semilogy(SNR_dBs, res_no_shadow[L], '-o', color=c, label=f"Céu Aberto (L={L})")
    
    # Shadowing (linha tracejada, x)
    plt.semilogy(SNR_dBs, res_heavy_shadow[L], '--x', color=c, label=f"Bloqueio 6dB (L={L})")

plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.xlabel("SNR (dB)", fontweight='bold')
plt.ylabel("BER", fontweight='bold')
plt.title("Impacto do Sombreamento (Shadowing) no Desempenho LoRa\n(Canal Rayleigh + Lognormal | Técnica MRC)")

plt.ylim(1e-5, 1)
plt.xlim(SNR_dBs[0], SNR_dBs[-1])

# Dividi a legenda em 2 colunas para não ocupar muito espaço no gráfico
plt.legend(ncol=2, loc="upper right") 
plt.tight_layout()

plt.savefig("shadowing.png", dpi=300)
print(f"Gráfico salvo em 'shadowing.png'")
plt.show()