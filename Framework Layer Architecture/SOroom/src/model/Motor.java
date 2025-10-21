package model;

public class Motor extends Kendaraan {

	public Motor(String nama, int harga) {
		super(nama, harga);
	}

	@Override
	public String getTipe() {
		return "Motor";
	}	

}
