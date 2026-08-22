// Reference dump: drives the ten-vad C front end (STFT + pitch estimator)
// over a 16 kHz mono 16-bit WAV and prints one line per hop:
//     frameIdx  pitchFreq  voiced
//
// Reproduces AUP_Aed_procAudio's pitch branch exactly:
//   * pre-emphasis feeds the STFT branch only,
//   * the pitch estimator reads the raw hop and the un-normalized bin power.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include "stft.h"
#include "pitch_est.h"

extern const float AUP_AED_STFTWindow_Hann768[768];

static bool read_wav_i16(const char* path, std::vector<int16_t>& out, int& sr, int& ch) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  char riff[12];
  if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) || memcmp(riff + 8, "WAVE", 4)) {
    fclose(f); return false;
  }
  int bits = 0; sr = 0; ch = 0;
  while (true) {
    char id[4]; uint32_t sz;
    if (fread(id, 1, 4, f) != 4) break;
    if (fread(&sz, 4, 1, f) != 1) break;
    if (!memcmp(id, "fmt ", 4)) {
      uint16_t fmt, nch, bps; uint32_t rate, brate; uint16_t align;
      fread(&fmt, 2, 1, f); fread(&nch, 2, 1, f); fread(&rate, 4, 1, f);
      fread(&brate, 4, 1, f); fread(&align, 2, 1, f); fread(&bps, 2, 1, f);
      ch = nch; sr = (int)rate; bits = bps;
      if (sz > 16) fseek(f, (long)sz - 16, SEEK_CUR);
    } else if (!memcmp(id, "data", 4)) {
      if (bits != 16) { fclose(f); return false; }
      out.resize(sz / 2);
      fread(out.data(), 1, sz, f);
      fclose(f);
      return true;
    } else {
      fseek(f, (long)sz + (sz & 1), SEEK_CUR);
    }
  }
  fclose(f);
  return false;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: dump_pitch <wav>\n"); return 1; }

  std::vector<int16_t> pcm; int sr = 0, ch = 0;
  if (!read_wav_i16(argv[1], pcm, sr, ch)) { fprintf(stderr, "bad wav\n"); return 1; }
  if (sr != 16000 || ch != 1) { fprintf(stderr, "need 16k mono, got %d/%d\n", sr, ch); return 1; }

  const int HOP = 256, FFT = 1024, WIN = 768, NBINS = FFT / 2 + 1;

  void* analyzer = NULL;
  if (AUP_Analyzer_create(&analyzer) < 0) return 1;
  Analyzer_StaticCfg acfg;
  AUP_Analyzer_getStaticCfg(analyzer, &acfg);
  acfg.win_len = WIN; acfg.hop_size = HOP; acfg.fft_size = FFT;
  acfg.ana_win_coeff = AUP_AED_STFTWindow_Hann768;
  if (AUP_Analyzer_memAllocate(analyzer, &acfg) < 0) return 1;
  if (AUP_Analyzer_init(analyzer) < 0) return 1;

  void* pe = NULL;
  if (AUP_PE_create(&pe) < 0) return 1;
  PE_StaticCfg pcfg;
  AUP_PE_getStaticCfg(pe, &pcfg);
  pcfg.fftSz = FFT; pcfg.anaWindowSz = WIN; pcfg.hopSz = HOP;
  pcfg.useLPCPreFiltering = 1; pcfg.procFs = 4000;
  if (AUP_PE_memAllocate(pe, &pcfg) < 0) return 1;
  if (AUP_PE_init(pe) < 0) return 1;
  PE_DynamCfg dcfg; AUP_PE_getDynamCfg(pe, &dcfg);
  dcfg.voicedThr = 0.4f;
  AUP_PE_setDynamCfg(pe, &dcfg);

  std::vector<float> raw(HOP), emph(HOP), spec(FFT), binPow(NBINS);
  float pre = 0.0f;
  size_t nFrames = pcm.size() / HOP;

  for (size_t fr = 0; fr < nFrames; fr++) {
    for (int i = 0; i < HOP; i++) {
      float x = (float)pcm[fr * HOP + i];
      raw[i] = x;
      emph[i] = x - 0.97f * pre;
      pre = x;
    }

    Analyzer_InputData ain; ain.input = emph.data(); ain.iLength = HOP;
    Analyzer_OutputData aout; aout.output = spec.data(); aout.oLength = FFT;
    if (AUP_Analyzer_proc(analyzer, &ain, &aout) < 0) return 1;

    // FFTW half-complex unpack, matching AUP_Aed_CalcBinPow.
    binPow[0] = spec[0] * spec[0];
    binPow[NBINS - 1] = spec[1] * spec[1];
    for (int i = 1; i < NBINS - 1; i++) {
      binPow[i] = spec[2 * i] * spec[2 * i] + spec[2 * i + 1] * spec[2 * i + 1];
    }

    PE_InputData pin;
    pin.timeSignal = raw.data(); pin.hopSz = HOP;
    pin.inBinPow = binPow.data(); pin.nBins = NBINS;
    PE_OutputData pout = {0, 0};
    if (AUP_PE_proc(pe, &pin, &pout) < 0) return 1;

    printf("%zu %.9g %d\n", fr, pout.pitchFreq, pout.voiced);
  }

  AUP_PE_destroy(&pe);
  AUP_Analyzer_destroy(&analyzer);
  return 0;
}
