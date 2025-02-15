package model;

public class Mobil extends Kendaraan {

	public Mobil(String merk) {
		super(merk);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void info() {
		System.out.println("Merek " +getMerk());
		
	}
	
}
