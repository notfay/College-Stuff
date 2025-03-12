package main;

import proto.Kendaraan;

public class Main {

	public static void main(String[] args) {
		Kendaraan mobil1 = new Kendaraan("Tesla", "Hitam");
		Kendaraan mobil2 = mobil1.clone();
		
		mobil2.gantiMerk("Xenia");
		
		mobil1.getInfo();
		mobil2.getInfo();
	}

}
