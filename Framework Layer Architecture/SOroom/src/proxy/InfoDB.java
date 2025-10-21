package proxy;

import model.Kendaraan;

public class InfoDB implements InfoKendaraan {

	@Override
	public String getInfo(Kendaraan kendaraan) {
		return kendaraan.getNama() + kendaraan.getHarga() + kendaraan.getTipe();
	}

}
