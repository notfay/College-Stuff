package abstarctfactory;

import model.Kendaraan;
import model.Motor;

public class MotorFactory implements FactoryKendaraan {

	@Override
	public Kendaraan produksiKendaraan() {
		System.out.println("Produksi Motor");
		return new Motor();
	}
	
	
}
