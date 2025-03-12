package adapter;

import model.Kendaraan;

public class KendaraanAdapter {
	public static String formatKendaraan(Kendaraan kendaraan) {
		return String.format("ID : %s, Merek: %s, Tahun: %d, Tipe: %s", 
				kendaraan.getId(),
				kendaraan.getMerek(),
				kendaraan.getTahun(),
				kendaraan.getTipe()
		);
	}
}
