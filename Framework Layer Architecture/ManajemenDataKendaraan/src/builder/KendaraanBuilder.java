package builder;

import model.Kendaraan;
import model.Mobil;
import model.Motor;

public class KendaraanBuilder {
	private String id;
	private String merek;
	private int tahun;
	private String tipe;
	
	public KendaraanBuilder setId(String id) {
		this.id = id;
		return this;
	}
	
	public KendaraanBuilder setMerek(String merek) {
		this.merek = merek;
		return this;
	}
	
	public KendaraanBuilder setTahun(int tahun) {
		this.tahun = tahun;
		return this;
	}
	
	public KendaraanBuilder setTipe(String tipe) {
		this.tipe = tipe;
		return this;
	}
	
	public Kendaraan produksi() {
		if("Mobil".equalsIgnoreCase(tipe)) {
			return new Mobil(id, merek, tahun);
		} else if("Motor".equalsIgnoreCase(tipe)) {
			return new Motor(id, merek, tahun);
		} else {
			System.out.println("Model kendaraan tidak tersedia!!!!");
			return null;
		}
	}

}
