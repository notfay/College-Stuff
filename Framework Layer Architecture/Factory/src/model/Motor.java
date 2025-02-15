package model;

public class Motor extends Kendaraan{

	public Motor(String merk) {
		super(merk);
	}

	@Override
	public void info() {
		System.out.println("Merek " + getMerk());
	}

}
