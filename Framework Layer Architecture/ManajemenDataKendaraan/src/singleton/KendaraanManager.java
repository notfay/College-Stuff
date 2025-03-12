package singleton;

import java.util.ArrayList;

import builder.KendaraanBuilder;
import model.Kendaraan;

public class KendaraanManager {
	private static KendaraanManager instance;
//	Dynamic Array
	private ArrayList<Kendaraan> kendaraans;
	private int totalId = 1;
	
//	Inisialisasi buat dynamic array
	private KendaraanManager() {
		kendaraans = new ArrayList<>();
	}
	
	public static KendaraanManager getInstance() {
		if(instance == null) {
			synchronized(KendaraanManager .class) {
				if(instance == null) {
					instance = new KendaraanManager();
				}
			}
		}
		
		return instance;
	}
	
	public void tambahKendaraan(String merek, int tahun, String tipe) {
		String id = "KE" + totalId++;
		Kendaraan kendaraan = new KendaraanBuilder()
				.setId(id)
				.setMerek(merek)
				.setTahun(tahun)
				.setTipe(tipe)
				.produksi();
		kendaraans.add(kendaraan);
	}
	
	public ArrayList<Kendaraan> ambilKendaraan() {
		return kendaraans;
	}
}
