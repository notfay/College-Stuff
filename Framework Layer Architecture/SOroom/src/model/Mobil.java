package model;

public class Mobil extends Kendaraan {

	public Mobil(String nama, int harga) {
		super(nama, harga);
	}

	@Override
	public String getTipe() {
		return "Mobil";
	}
	
}
