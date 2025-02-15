package factory;

import model.*;

public class PabrikKendaraan {
	public static Kendaraan produksiKendaraan(String tipe) {
		if(tipe.equalsIgnoreCase("Motor")) {
			return new Motor("Honda");
		} 
		else if (tipe.equalsIgnoreCase("Mobil")) {
			return new Mobil ("Tesla");
		}
		else {
			System.out.println("Kendaraan tidak diproduksi");
			return null;
		}
	}
}
