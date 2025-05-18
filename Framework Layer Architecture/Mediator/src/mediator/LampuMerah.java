package mediator;

public class LampuMerah implements Mediator{
	
	
	private boolean hijau = true;
	

	@Override
	public void mauNyebrang(String namaMobil) {
		if (hijau) {
			System.out.println(namaMobil + " jalan");
			hijau = false;
		} else {
			System.out.println(namaMobil + " berhenti");
			hijau = true;
		}
		
	}
	
}
