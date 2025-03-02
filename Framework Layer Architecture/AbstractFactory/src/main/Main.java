package main;

import abstarctfactory.FactoryKendaraan;
import abstarctfactory.MobilFactory;
import abstarctfactory.MotorFactory;
import model.Kendaraan;

public class Main {

	public static void main(String[] args) {
		FactoryKendaraan mobilFactoryKendaraan = new MobilFactory();
		Kendaraan mobil = mobilFactoryKendaraan.produksiKendaraan();
		mobil.displayMerek();
		
		FactoryKendaraan motorFactoryKendaraan = new MotorFactory();
		Kendaraan motor = motorFactoryKendaraan.produksiKendaraan();
		motor.displayMerek();
	}

}
