package main;


import factory.PabrikKendaraan;
import model.Kendaraan;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Kendaraan motor = PabrikKendaraan.produksiKendaraan("motor");
		motor.info();
		
		
		model.Kendaraan mobil = PabrikKendaraan.produksiKendaraan("Mobil");
		mobil.info();
		
		Kendaraan sepeda = PabrikKendaraan.produksiKendaraan("sepeda");
	}

}
