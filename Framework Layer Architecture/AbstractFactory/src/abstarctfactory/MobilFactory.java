package abstarctfactory;

import model.Kendaraan;
import model.Mobil;

public class MobilFactory implements FactoryKendaraan{

	@Override
	public Kendaraan produksiKendaraan() {
		System.out.println("Produksi Mobil");
		return new Mobil();
	}


}
